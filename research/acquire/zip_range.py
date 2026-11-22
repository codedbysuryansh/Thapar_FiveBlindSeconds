#!/usr/bin/env python3
"""Range-request reader for remote ZIP archives (incl. ZIP64 and stored nested zips). Research instrument.
Lists the central directory and extracts chosen small members without downloading imagery."""
import struct, zlib, urllib.request, time, os, json, hashlib, sys
UA={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15"}
import urllib.error
_last=[0.0]
def _open(req,timeout=120):
    """Polite opener: >=2 s between requests, honours HTTP 429 with Retry-After/back-off (rate limits are respected, never bypassed)."""
    for k in range(8):
        w=2.0-(time.time()-_last[0])
        if w>0: time.sleep(w)
        _last[0]=time.time()
        try: return urllib.request.urlopen(req,timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code not in (429,503): raise
            ra=e.headers.get("Retry-After"); time.sleep(min(600,int(ra)) if ra and ra.isdigit() else 45*(k+1))
    raise IOError("rate-limited repeatedly")
class Remote:
    def __init__(s,url,base=0,length=None):
        s.url=url; s.base=base
        if length is None:
            r=_open(urllib.request.Request(url,headers={**UA,"Range":"bytes=0-0"}),timeout=90)
            cr=r.headers.get("Content-Range"); r.read()
            if not cr: raise RuntimeError("server ignored Range; status %s"%r.status)
            length=int(cr.split("/")[1])
        s.length=length
    def read(s,a,n):
        if n<=0: return b""
        h={**UA,"Range":"bytes=%d-%d"%(s.base+a,s.base+a+n-1)}
        for k in range(6):
            try:
                with _open(urllib.request.Request(s.url,headers=h),timeout=180) as r:
                    d=r.read()
                    if len(d)==n: return d
                    raise IOError("short read %d/%d status %s"%(len(d),n,r.status))
            except Exception as e:
                err=e; time.sleep(2*(k+1))
        raise err
def entries(rf):
    tail_n=min(rf.length,1<<16); tail=rf.read(rf.length-tail_n,tail_n)
    i=tail.rfind(b"PK\x05\x06")
    if i<0: raise RuntimeError("no EOCD")
    _,_,_,_,n,cdsize,cdoff,_=struct.unpack("<IHHHHIIH",tail[i:i+22])
    j=tail.rfind(b"PK\x06\x07")
    if j>=0 and (cdoff==0xFFFFFFFF or n==0xFFFF or cdsize==0xFFFFFFFF):
        _,_,eocd64,_=struct.unpack("<IIQI",tail[j:j+20])
        e=rf.read(eocd64,56); n,cdsize,cdoff=struct.unpack("<QQQ",e[32:56])
    cd=rf.read(cdoff,cdsize); out=[]; p=0
    while p+46<=len(cd) and cd[p:p+4]==b"PK\x01\x02":
        (_,_,_,flag,meth,_,_,crc,csz,usz,nl,el,cl,_,_,_,lho)=struct.unpack("<IHHHHHHIIIHHHHHII",cd[p:p+46])
        name=cd[p+46:p+46+nl].decode("utf-8","replace"); ex=cd[p+46+nl:p+46+nl+el]; q=0
        while q+4<=len(ex):
            hid,hs=struct.unpack("<HH",ex[q:q+4]); body=ex[q+4:q+4+hs]
            if hid==1:
                r=0
                if usz==0xFFFFFFFF: usz=struct.unpack("<Q",body[r:r+8])[0]; r+=8
                if csz==0xFFFFFFFF: csz=struct.unpack("<Q",body[r:r+8])[0]; r+=8
                if lho==0xFFFFFFFF: lho=struct.unpack("<Q",body[r:r+8])[0]; r+=8
            q+=4+hs
        out.append({"name":name,"method":meth,"csize":csz,"usize":usz,"lho":lho,"crc":crc})
        p+=46+nl+el+cl
    return out
def data_offset(rf,e):
    h=rf.read(e["lho"],30); nl,el=struct.unpack("<HH",h[26:30]); return e["lho"]+30+nl+el
def extract(rf,e):
    raw=rf.read(data_offset(rf,e),e["csize"])
    d=raw if e["method"]==0 else zlib.decompress(raw,-15)
    assert zlib.crc32(d)&0xFFFFFFFF==e["crc"],"CRC mismatch "+e["name"]
    return d
def nested(rf,e):
    assert e["method"]==0,"nested zip is compressed; cannot range-read"
    return Remote(rf.url,rf.base+data_offset(rf,e),e["usize"])
if __name__=="__main__":
    url=sys.argv[2]; rf=Remote(url); es=entries(rf)
    if sys.argv[1]=="ls":
        print("archive bytes",rf.length,"entries",len(es))
        from collections import Counter
        c=Counter(os.path.splitext(x["name"])[1] for x in es); print(dict(c))
        for x in es:
            if not x["name"].lower().endswith((".png",".jpg",".jpeg","/")): print(x["name"],x["method"],x["csize"],x["usize"])

#!/usr/bin/env python3
"""List EuRoC bitstreams on the ETH Research Collection (DSpace REST). Research instrument."""
import json, urllib.request, sys
UA={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15","Accept":"application/json"}
B="https://www.research-collection.ethz.ch/server/api"
def get(u):
    with urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=60) as r: return json.load(r)
item=get(B+"/core/items/bcaf173e-5dac-484b-bc37-faf97a594f1f")
md=item["metadata"]
for k,v in md.items():
    if any(s in k for s in ("rights","license","date.issued","identifier.doi","description.abstract","title")):
        print(k,"=>"," | ".join(str(x["value"])[:400] for x in v))
out=[]
bundles=get(B+"/core/items/"+item["uuid"]+"/bundles?size=50")
for b in bundles["_embedded"]["bundles"]:
    page=0
    while True:
        bs=get(B+f"/core/bundles/{b['uuid']}/bitstreams?size=100&page={page}")
        for s in bs["_embedded"]["bitstreams"]:
            out.append({"bundle":b["name"],"name":s["name"],"bytes":s["sizeBytes"],"uuid":s["uuid"],"checksum":s.get("checkSum"),"content":s["_links"]["content"]["href"]})
        if page+1>=bs["page"]["totalPages"]: break
        page+=1
json.dump(out,open("data/euroc/BITSTREAMS.json","w"),indent=1)
for o in out: print(o["bundle"],o["name"],o["bytes"],o["uuid"],(o["checksum"] or {}).get("value"))

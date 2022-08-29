
set session  join_reordering_strategy ='none';

with nations as (select n.nationkey, n.name from 
local_pnb2dw1.oerling_nation_3k_nz as n,
	local_pnb2dw1.oerling_region_3k_nz as r where r.regionkey = n.regionkey and r.name = 'ASIA'),
suppliers_from_nation as (select s.suppkey, n.nationkey, n.name from 
local_pnb2dw1.oerling_supplier_3k_nz  s,
	       nations n
   where n.nationkey = s.nationkey),
     cust_from_nation as (select c.custkey, ns.nationkey from
local_pnb2dw1.oerling_customer_3k_nz as c, ns
  where ns.nationkey = c.nationkey)
select
	sn.name,
	sum(l.extendedprice * (1 - l.discount)) as revenue
from	local_pnb2dw1.oerling_lineitem_3k_nz as l,
	   supplier_from_nation sn,
   (select o.orderkey, cn.nationkey from
    local_pnb2dw1.oerling_orders_3k_nz as o,
    				       cust_from_nation cn
     where cn.custkey = o.custkey         and o.orderdate >= '1994-01-01'
        and o.orderdate <  '1995-01-01' 
) ocn
   where
 l.orderkey = ocn.orderkey
	and l.suppkey = sn.suppkey
	and sn.nationkey = ocn.nationkey

group by
	sn.name
order by
	revenue desc;

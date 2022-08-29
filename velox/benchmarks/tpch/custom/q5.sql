-- TPC-H/TPC-R Local Supplier Volume Query (Q5)
-- Functional Query Definition
-- Approved February 1998
select
	n.name,
	sum(l.extendedprice * (1 - l.discount)) as revenue
from
	local_pnb2dw1.oerling_customer_3k_nz as c,
	local_pnb2dw1.oerling_orders_3k_nz as o,
	local_pnb2dw1.oerling_lineitem_3k_nz as l,
	local_pnb2dw1.oerling_supplier_3k_nz as s,
	local_pnb2dw1.oerling_nation_3k_nz as n,
	local_pnb2dw1.oerling_region_3k_nz as r
where
	c.custkey = o.custkey
	and l.orderkey = o.orderkey
	and l.suppkey = s.suppkey
	and c.nationkey = s.nationkey
	and s.nationkey = n.nationkey
	and n.regionkey = r.regionkey
	and r.name = 'ASIA'
        and o.orderdate >= '1994-01-01'
        and o.orderdate <  '1995-01-01' 
group by
	n.name
order by
	revenue desc;

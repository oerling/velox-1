-- TPC-H/TPC-R Shipping Priority Query (Q3)
-- Functional Query Definition
-- Approved February 1998
select
	l.orderkey,
	sum(l.extendedprice * (1 - l.discount)) as revenue,
	o.orderdate,
	o.shippriority
from
	local_pnb2dw1.oerling_lineitem_3k_nz as l,
	local_pnb2dw1.oerling_orders_3k_nz as o,
	local_pnb2dw1.oerling_customer_3k_nz as c
where
	c.mktsegment = 'BUILDING'
	and c.custkey = o.custkey
	and l.orderkey = o.orderkey
	and o.orderdate <  '1995-03-15'
	and l.shipdate >  '1995-03-15'
group by
	l.orderkey,
	o.orderdate,
	o.shippriority
order by
	revenue desc,
	o.orderdate
limit 10;

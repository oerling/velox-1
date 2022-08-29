

select
	l.orderkey,
	sum(l.extendedprice * (1 - l.discount)) as revenue,
	co.orderdate,
	co.shippriority
from
	local_pnb2dw1.oerling_lineitem_3k_nz as l,
  (select 
    o.orderkey, o.orderdate, o.shippriority
   from local_pnb2dw1.oerling_orders_3k_nz as o,
       local_pnb2dw1.oerling_customer_3k_nz as c
          where c.custkey = o.custkey 
	and o.orderdate <  '1995-03-15'
  and 	c.mktsegment = 'BUILDING') co

where
	l.orderkey = co.orderkey
	and l.shipdate >  '1995-03-15'
group by
	l.orderkey,
	co.orderdate,
	co.shippriority
order by
	revenue desc,
	co.orderdate
limit 10;


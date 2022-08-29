-- TPC-H/TPC-R Customer Distribution Query (Q13)
-- Functional Query Definition
-- Approved February 1998
select
	c_count,
	count(*) as custdist
from
	(
		select
			c.custkey,
			count(o.orderkey) c_count
		from
local_pnb2dw1.oerling_orders_3k_nz o
right join local_pnb2dw1.oerling_customer_3k_nz c   on
				c.custkey = o.custkey
				and o.comment not like '%special%requests%'
		group by
			c.custkey
	) as cust
group by
	c_count
order by
	custdist desc,
	c_count desc;

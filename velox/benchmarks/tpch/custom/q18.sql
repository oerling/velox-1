-- TPC-H/TPC-R Large Volume Customer Query (Q18)
-- Function Query Definition
-- Approved February 1998
select
	c.name,
	c.custkey,
	o.orderkey,
	o.orderdate,
	o.totalprice,
	sum(l.quantity)
from
	local_pnb2dw1.oerling_customer_3k_nz as c,
	local_pnb2dw1.oerling_orders_3k_nz as o,
	local_pnb2dw1.oerling_lineitem_3k_nz as l
where
	o.orderkey in (
		select
			orderkey
		from
			local_pnb2dw1.oerling_lineitem_3k_nz
		group by
			orderkey
		having
            sum(quantity) > 300
	)
	and c.custkey = o.custkey
	and o.orderkey = l.orderkey
group by
	c.name,
	c.custkey,
	o.orderkey,
	o.orderdate,
	o.totalprice
order by
	o.totalprice desc,
	o.orderdate
limit 100;

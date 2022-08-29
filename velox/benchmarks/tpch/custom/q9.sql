-- TPC-H/TPC-R Product Type Profit Measure Query (Q9)
-- Functional Query Definition
-- Approved February 1998
select
		nation,
	o_year,
	sum(amount) as sum_profit
from
	(
		select
			n.name as nation,
			extract(year from date(o.orderdate)) as o_year,
			l.extendedprice * (1 - l.discount) - ps.supplycost * l.quantity as amount
		from
			local_pnb2dw1.oerling_lineitem_3k_nz as l,
			local_pnb2dw1.oerling_part_3k_nz as p,
			local_pnb2dw1.oerling_partsupp_3k_nz as ps,
			local_pnb2dw1.oerling_orders_3k_nz as o,
			local_pnb2dw1.oerling_supplier_3k_nz as s,
			local_pnb2dw1.oerling_nation_3k_nz as n
		where
			s.suppkey = l.suppkey
			and ps.suppkey = l.suppkey
			and ps.partkey = l.partkey
			and p.partkey = l.partkey
			and o.orderkey = l.orderkey
			and s.nationkey = n.nationkey
			and p.name like '%green%'
	) as profit
group by
	nation,
	o_year
order by
	nation,
	o_year desc;

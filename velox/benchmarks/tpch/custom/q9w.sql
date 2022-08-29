

with parts as (select partkey from
local_pnb2dw1.oerling_part_3k_nz  where   name like '%green%'),
  ps as (select ps.partkey, ps.suppkey, ps.supplycost from
  local_pnb2dw1.oerling_partsupp_3k_nz ps, parts p where ps.partkey = p.partkey),
    sn as (select s.suppkey, n.name from
			local_pnb2dw1.oerling_supplier_3k_nz as s,
			local_pnb2dw1.oerling_nation_3k_nz as n

where s.nationkey = n.nationkey)


select
		nation,
	o_year,
	sum(amount) as sum_profit
from
	(
		select
			sn.name as nation,
			substr(o.orderdate, 1, 4) as o_year,
			l.extendedprice * (1 - l.discount) - ps.supplycost * l.quantity as amount
		from
			local_pnb2dw1.oerling_lineitem_3k_nz as l,
			
  ps,
			local_pnb2dw1.oerling_orders_3k_nz as o,
			  sn
		where
			sn.suppkey = l.suppkey
			and ps.suppkey = l.suppkey
			and ps.partkey = l.partkey
			and o.orderkey = l.orderkey

	) as profit
group by
	nation,
	o_year
order by
	nation,
	o_year desc;

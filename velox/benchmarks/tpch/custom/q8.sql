-- TPC-H/TPC-R National Market Share Query (Q8)
-- Functional Query Definition
-- Approved February 1998
select
	o_year,
	sum(case
		when nation = 'BRAZIL' then volume
		else 0
	end) / sum(volume) as mkt_share
from
	(
		select
			extract(year from date(o.orderdate)) as o_year,
			l.extendedprice * (1 - l.discount) as volume,
			n2.name as nation
		from
			local_pnb2dw1.oerling_part_3k_nz as p,
			local_pnb2dw1.oerling_supplier_3k_nz as s,
			local_pnb2dw1.oerling_lineitem_3k_nz as l,
			local_pnb2dw1.oerling_orders_3k_nz as o,
			local_pnb2dw1.oerling_customer_3k_nz as c,
			local_pnb2dw1.oerling_nation_3k_nz as n1,
			local_pnb2dw1.oerling_nation_3k_nz as n2,
			local_pnb2dw1.oerling_region_3k_nz as r
		where
			p.partkey = l.partkey
			and s.suppkey = l.suppkey
			and l.orderkey = o.orderkey
			and o.custkey = c.custkey
			and c.nationkey = n1.nationkey
						and n1.regionkey = r.regionkey
			and r.name = 'AMERICA'
			and s.nationkey = n2.nationkey
			and o.orderdate between date '1995-01-01' and date '1996-12-31'
			and p.type = 'ECONOMY ANODIZED STEEL'
	) as all_local_nations
group by
	o_year
order by
	o_year;

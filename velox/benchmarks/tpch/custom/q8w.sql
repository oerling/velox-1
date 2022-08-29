-- TPC-H/TPC-R National Market Share Query (Q8)
-- Functional Query Definition
-- Approved February 1998
select
	o_year,
	sum(case
		when local_pnb2dw1.oerling_nation_3k_nz = 'BRAZIL' then volume
		else 0
	end) / sum(volume) as mkt_share
from
	(
		select
			substr(o.orderdate, 4) as o_year,
			l.extendedprice * (1 - l.discount) as volume,
			n2.name as nationkey from
			local_pnb2dw1.oerling_lineitem_3k_nz as l,
			local_pnb2dw1.oerling_part_3k_nz as p,
(select  o.orderkey, name n1 from 
local_pnb2dw1.oerling_orders_3k_nz as o,
  select c.custkey, name from 
local_pnb2dw1.oerling_customer_3k_nz as c,
   (select nationkey from
   

			local_pnb2dw1.oerling_supplier_3k_nz as s,
			local_pnb2dw1.oerling_nation_3k_nz as n1,
			local_pnb2dw1.oerling_nation_3k_nz as n2,
			local_pnb2dw1.oerling_region_3k_nz as r
		where
			p.nationkey = l.partkey
			and s.suppkey = l.suppkey
			and l.orderkey = o.orderkey
			and o.custkey = c.custkey
			and c.nationkey = n1.nationkey
			and n1.nationkey = r.regionkey
			and r.name = 'AMERICA'
			and s.nationkey = n2.nationkey
			and o.orderdate between '1995-01-01' and '1996-12-31'
			and p.type = 'ECONOMY ANODIZED STEEL'
	) as all_local_pnb2dw1.oerling_nation_3k_nzs
group by
	o_year
order by
	o_year;

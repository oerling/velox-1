-- TPC-H/TPC-R Parts/Supplier Relationship Query (Q16)
-- Functional Query Definition
-- Approved February 1998

set session join_distribution_type = 'automatic';

select
	p.brand,
	p.type,
	p.size,
	approx_distinct(ps.suppkey) as supplier_cnt
from
	local_pnb2dw1.oerling_partsupp_3k_nz as ps,
	local_pnb2dw1.oerling_part_3k_nz as p
where
		p.partkey = ps.partkey
	and p.brand <> 'Brand#45'
	and p.type not like 'MEDIUM POLISHED%'
	and p.size in (49, 14, 23, 45, 19, 3, 36, 9)
	and ps.suppkey not in (
		select
			suppkey
		from
			local_pnb2dw1.oerling_supplier_3k_nz
		where
			comment like '%Customer%Complaints%'
	)
group by
	p.brand,
	p.type,
	p.size
order by
      	supplier_cnt desc,
	p.brand,
	p.type,
	p.size;

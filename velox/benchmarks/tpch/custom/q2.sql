-- TPC-H/TPC-R Minimum Cost Supplier Query (Q2)
-- Functional Query Definition
-- Approved February 1998
select
	s.acctbal,
	s.name,
	n.name,
		p.partkey,
	p.mfgr,
	s.address,
	s.phone,
	s.comment
from
	local_pnb2dw1.oerling_part_3k_nz as p,
	local_pnb2dw1.oerling_supplier_3k_nz as s,
	local_pnb2dw1.oerling_partsupp_3k_nz as ps,
	local_pnb2dw1.oerling_nation_3k_nz as n,
	local_pnb2dw1.oerling_region_3k_nz as r
where
		p.partkey = ps.partkey
	and s.suppkey = ps.suppkey
	and p.size = 15
	and p.type like '%BRASS'
	and s.nationkey = n.nationkey
	and n.nationkey = r.regionkey
	and r.name = 'EUROPE'
	and ps.supplycost = (
		select
			min(ps.supplycost)
		from
			local_pnb2dw1.oerling_partsupp_3k_nz as ps,
			local_pnb2dw1.oerling_supplier_3k_nz as s,
			local_pnb2dw1.oerling_nation_3k_nz as n,
			local_pnb2dw1.oerling_region_3k_nz as r
		where
			p.partkey = ps.partkey
			and s.suppkey = ps.suppkey
			and s.nationkey = n.nationkey
						and n.regionkey = r.regionkey
			and r.name = 'EUROPE'
	)
order by
	s.acctbal desc,
	n.name,
	s.name,
		p.partkey
limit 100;

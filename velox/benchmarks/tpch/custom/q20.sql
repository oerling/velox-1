-- TPC-H/TPC-R Potential Part Promotion Query (Q20)
-- Function Query Definition
-- Approved February 1998
select
	s.name,
	s.address
from
	local_pnb2dw1.oerling_supplier_3k_nz as s,
	local_pnb2dw1.oerling_nation_3k_nz as n
where
	s.suppkey in (
		select
            ps.suppkey
		from
			local_pnb2dw1.oerling_partsupp_3k_nz as ps
		where
			ps.partkey in (
				select
					partkey
				from
					local_pnb2dw1.oerling_part_3k_nz
				where
					name like 'forest%'
			)
			and ps.availqty > (
				select
					0.5 * sum(l.quantity)
				from
					local_pnb2dw1.oerling_lineitem_3k_nz as l
				where
					l.partkey = ps.partkey
					and l.suppkey = ps.suppkey
					and l.shipdate >= date('1994-01-01')
					and l.shipdate < date('1994-01-01') + interval '1' year
			)
	)
	and s.nationkey = n.nationkey
	and n.name = 'CANADA'
order by
	s.name limit 100;

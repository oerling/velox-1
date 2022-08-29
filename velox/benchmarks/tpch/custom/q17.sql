-- TPC-H/TPC-R Small-Quantity-Order Revenue Query (Q17)
-- Functional Query Definition
-- Approved February 1998
select
	sum(l.extendedprice) / 7.0 as avg_yearly
from
	local_pnb2dw1.oerling_lineitem_3k_nz as l,
	local_pnb2dw1.oerling_part_3k_nz as p
where
	p.local_pnb2dw1.oerling_part_3k_nzkey = l.partkey
	and p.brand = 'Brand#23'
	and p.container = 'MED BOX'
	and l.quantity < (
		select
			0.2 * avg(quantity)
		from
			local_pnb2dw1.oerling_lineitem_3k_nz
		where
			local_pnb2dw1.oerling_part_3k_nzkey = p.partkey
	);

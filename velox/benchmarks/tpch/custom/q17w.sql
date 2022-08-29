-- TPC-H/TPC-R Small-Quantity-Order Revenue Query (Q17)
-- Functional Query Definition
-- Approved February 1998
select
	sum(l.extendedprice) / 7.0 as avg_yearly
from
	local_pnb2dw1.oerling_lineitem_3k_nz as l,
	local_pnb2dw1.oerling_part_3k_nz as p
where
	p.partkey = l.partkey
	and p.brand = 'Brand#23'
	and p.container = 'MED BOX'
	and l.quantity < (
		select
			0.2 * avg(quantity)
		from
			local_pnb2dw1.oerling_lineitem_3k_nz l2,
			local_pnb2dw1.oerling_part_3k_nz p2
		where
	p2.partkey = l2.partkey
	and p2.brand = 'Brand#23'
	and p2.container = 'MED BOX'
	and l2.partkey = p.partkey
	);

-- TPC-H/TPC-R Suppliers Who Kept Orders Waiting Query (Q21)
-- Functional Query Definition
-- Approved February 1998
select
	s.name,
	count(*) as numwait
from
	local_pnb2dw1.oerling_supplier_3k_nz as s,
	local_pnb2dw1.oerling_lineitem_3k_nz as l1,
	local_pnb2dw1.oerling_orders_3k_nz as o,
	local_pnb2dw1.oerling_nation_3k_nz as n
where
	s.suppkey = l1.suppkey
	and o.orderkey = l1.orderkey
	and o.orderstatus = 'F'
	and l1.receiptdate > l1.commitdate
	and exists (
		select
			*
		from
			local_pnb2dw1.oerling_lineitem_3k_nz l2,
			local_pnb2dw1.oerling_supplier_3k_nz s2,
			local_pnb2dw1.oerling_nation_3k_nz n2
		where
		l2.suppkey = s2.suppkey and s2.nationkey = n2.nationkey and n2.name = 'SAUDI ARABIA' and 
			l2.orderkey = l1.orderkey
			and l2.suppkey <> l1.suppkey
	)
	and not exists (
		select
			*
		from
			local_pnb2dw1.oerling_lineitem_3k_nz l3,
			local_pnb2dw1.oerling_supplier_3k_nz s3,
			local_pnb2dw1.oerling_nation_3k_nz n3
		where
		l3.suppkey = s3.suppkey and s3.nationkey = n3.nationkey and n3.name = 'SAUDI ARABIA' and 
l3.orderkey = l1.orderkey
			and l3.suppkey <> l1.suppkey
			and l3.receiptdate > l3.commitdate
	)
	and s.nationkey = n.nationkey
	and n.name = 'SAUDI ARABIA'
group by
	s.name
order by
	numwait desc,
	s.name
limit 100;

-- TPC-H/TPC-R Top Supplier Query (Q15)
-- Functional Query Definition
-- Approved February 1998

create view revenue as
	select
		suppkey as local_pnb2dw1.oerling_supplier_3k_nz_no,
		sum(extendedprice * (1 - discount)) as total_revenue
	from
		local_pnb2dw1.oerling_lineitem_3k_nz
	where
	        shipdate >= date '1996-01-01'
        	and  shipdate < date '1996-01-01' + interval '3' month
	group by
		suppkey;

select
	su.suppkey,
	su.name,
	su.address,
	su.phone,
	total_revenue
from
	local_pnb2dw1.oerling_supplier_3k_nz as su,
	revenue
where
	su.suppkey = local_pnb2dw1.oerling_supplier_3k_nz_no
	and total_revenue = (
		select
			max(total_revenue)
		from
			revenue
	)
order by
	su.suppkey;

drop view revenue;

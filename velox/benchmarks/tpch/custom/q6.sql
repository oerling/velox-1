-- TPC-H/TPC-R Forecasting Revenue Change Query (Q6)
-- Functional Query Definition
-- Approved February 1998
select
	sum(extendedprice * discount) as revenue
from
	local_pnb2dw1.oerling_lineitem_3k_nz
where
	shipdate >= '1994-01-01'
        and shipdate <  '1995-01-01'
        and discount between  0.06 -  0.01 and  0.06 +  0.01
	and quantity < 24;

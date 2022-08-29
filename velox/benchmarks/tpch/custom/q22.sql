-- TPC-H/TPC-R Global Sales Opportunity Query (Q22)
-- Functional Query Definition
-- Approved February 1998
select
	cntrycode,
	count(*) as numcust,
	sum(c_acctbal) as totacctbal
from
	(
		select
			substring(phone from 1 for 2) as cntrycode,
			acctbal as c_acctbal
		from
			local_pnb2dw1.oerling_customer_3k_nz
		where
			substring(phone from 1 for 2) in
                ('13', '31', '23', '29', '30', '18', '17')
			and acctbal > (
				select
					avg(acctbal)
				from
					local_pnb2dw1.oerling_customer_3k_nz
				where
					acctbal > 0.00
					and substring(phone from 1 for 2) in
                        ('13', '31', '23', '29', '30', '18', '17')
			)
			and not exists (
				select
					*
				from
					local_pnb2dw1.oerling_orders_3k_nz as o
				where
					o.custkey = custkey
			)
	) as custsale
group by
	cntrycode
order by
	cntrycode;

with cn as (select c.custkey, n.name from
			local_pnb2dw1.oerling_customer_3k_nz as c,
			local_pnb2dw1.oerling_nation_3k_nz as n
			  where c.nationkey = n.nationkey and n.name in ('FRANCE', 'GERMANY')),
  oc as (select o.orderkey, name from 
			local_pnb2dw1.oerling_orders_3k_nz as o, cn
			   where cn.custkey = o.custkey),

sn as (select s.suppkey, n.name from
			local_pnb2dw1.oerling_supplier_3k_nz as s,
			local_pnb2dw1.oerling_nation_3k_nz as n 
  where n.nationkey = s.nationkey and n.name in ('FRANCE', 'GERMANY'))

select
	supp_nation,
	cust_nation,
	l_year,
	sum(volume) as revenue
from
	(
		select				sn.name as supp_nation,
			oc.name as cust_nation,
			substr(l.shipdate, 1, 4) as l_year,
			l.extendedprice * (1 - l.discount) as volume
		from
			local_pnb2dw1.oerling_lineitem_3k_nz as l,
  oc, sn		where
			sn.suppkey = l.suppkey
			and oc.orderkey = l.orderkey

			and (
				(sn.name = 'FRANCE' and oc.name = 'GERMANY')
				or (sn.name = 'GERMANY' and oc.name = 'FRANCE')
			)
			and l.shipdate between '1995-01-01' and '1996-12-31'
	) as shipping
group by
	supp_nation,
	cust_nation,
	l_year
order by
	supp_nation,
	cust_nation,
	l_year;



***

with cn as (select c.custkey, n.name from
			local_pnb2dw1.oerling_customer_3k_nz as c,
			local_pnb2dw1.oerling_nation_3k_nz as n
			  where c.nationkey = n.nationkey and n.name in ('FRANCE', 'GERMANY')),
  oc as (select o.orderkey, name from 
			local_pnb2dw1.oerling_orders_3k_nz as o, cn
			   where cn.custkey = o.custkey)


select
	cust_nation,
	l_year,
	sum(volume) as revenue
from
	(select
			oc.name as cust_nation,
			substr(l.shipdate, 1, 4) as l_year,
			l.extendedprice * (1 - l.discount) as volume
		from
			local_pnb2dw1.oerling_lineitem_3k_nz as l,
  oc
  where
			oc.orderkey = l.orderkey
			and l.shipdate between '1995-01-01' and '1996-12-31'
	) as shipping
group by
	cust_nation,
	l_year
order by
	cust_nation,
	l_year;

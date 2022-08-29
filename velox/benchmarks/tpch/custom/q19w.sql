-- TPC-H/TPC-R Discounted Revenue Query (Q19)
-- Functional Query Definition
-- Approved February 1998

--set session join_distribution_type = 'broadcast';
select
	sum(l.extendedprice* (1 - l.discount)) as revenue
from
	local_pnb2dw1.oerling_lineitem_3k_nz as l,
  (select partkey, size, brand, container from 
local_pnb2dw1.oerling_part_3k_nz p
   where (p.brand = 'Brand#12'
		and p.container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
		and p.size between 1 and 5
)
  or  (p.brand = 'Brand#23'
		and p.container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
		and p.size between 1 and 10
)
  or ( 
		 p.size between 1 and 15
		and p.brand = 'Brand#34'
		and p.container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
)
) p
where
	(
		p.partkey = l.partkey
		and p.brand = 'Brand#12'
		and p.container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
		and l.quantity >= 1 and l.quantity <= 1 + 10
		and p.size between 1 and 5
		and l.shipmode in ('AIR', 'AIR REG')
		and l.shipinstruct = 'DELIVER IN PERSON'
	)
	or
	(
		p.partkey = l.partkey
		and p.brand = 'Brand#23'
		and p.container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
		and l.quantity >= 10 and l.quantity <= 10 + 10
		and p.size between 1 and 10
		and l.shipmode in ('AIR', 'AIR REG')
		and l.shipinstruct = 'DELIVER IN PERSON'
	)
	or
	(
		p.partkey = l.partkey
		and p.brand = 'Brand#34'
		and p.container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
		and l.quantity >= 20 and l.quantity <= 20 + 10
		and p.size between 1 and 15
		and l.shipmode in ('AIR', 'AIR REG')
		and l.shipinstruct = 'DELIVER IN PERSON'
	);

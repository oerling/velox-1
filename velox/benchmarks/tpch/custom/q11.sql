-- TPC-H/TPC-R Important Stock Identification Query (Q11)
-- Functional Query Definition
-- Approved February 1998
select
	ps.partkey,
	sum(ps.supplycost * ps.availqty) as value
from
	local_pnb2dw1.oerling_partsupp_3k_nz as ps,
	local_pnb2dw1.oerling_supplier_3k_nz as s,
	local_pnb2dw1.oerling_nation_3k_nz as n
where
	ps.suppkey = s.suppkey
	and s.nationkey = n.nationkey
	and n.name = 'GERMANY'
group by
	ps.partkey
having
    sum(ps.supplycost * ps.availqty) > (
        select
            sum(ps.supplycost * ps.availqty) * 0.0001
        from
            local_pnb2dw1.oerling_partsupp_3k_nz as ps,
            local_pnb2dw1.oerling_supplier_3k_nz as s,
            local_pnb2dw1.oerling_nation_3k_nz as n
        where
            ps.suppkey = s.suppkey
            and s.nationkey = n.nationkey
            and n.name = 'GERMANY'
    )
order by value desc;

set session join_reordering_strategy='automatic';set session max_drivers_per_task = 18;SET SESSION join_distribution_type='automatic';

set session prefer_partial_aggregation = false;

select partkey, checksum(comment || comment) from local_pnb2dw1.oerling_lineitem_3k_nz group by partkey limit 10;


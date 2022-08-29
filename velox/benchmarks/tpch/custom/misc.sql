
select partkey, checksum(comment || comment) from 
	local_pnb2dw1.oerling_lineitem_3k_nz
	group by partkey limit 10;
	

select partkey, checksum(if (linenumber = 6, comment || comment, null)) from 
	local_pnb2dw1.oerling_lineitem_3k_nz
	group by partkey limit 10;



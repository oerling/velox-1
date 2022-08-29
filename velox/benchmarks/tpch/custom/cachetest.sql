

set session max_drivers_per_task = 16;

select count(*), max(orderkey), max(linenumber), max(partkey), max(suppkey), max(extendedprice), max(quantity), max(tax), max(discount), max(length(shipdate)), max(length(commitdate)), max(length(receiptdate)), max(length(shipmode)), max(length(shipinstruct)), max(length(comment)) from local_pnb2dw1.oerling_lineitem_3k_nz where partkey between 100000000 and b110000000;


-- iow 19.5B 
-- load 53s 


select count(*), max(orderkey), max(linenumber), max(partkey), max(suppkey), max(extendedprice), max(quantity), max(tax), max(discount), max(length(shipdate)), max(length(commitdate)), max(length(receiptdate)), max(length(shipmode)), max(length(shipinstruct)), max(length(comment)) from local_pnb2dw1.oerling_lineitem_3k_nz where partkey between 200000000 and 300000000;


select count(*), max(orderkey + 1), max(linenumber + 1), max(partkey + 1), max(suppkey + 1), max(extendedprice * (1.0 - discount)), max(quantity + 1), max(if (quantity > 20, 0, tax + 1)),
  max(discount + 1), max(length(shipdate)), max(length(commitdate)), max(length(receiptdate)), max(length(shipmode)), max(length(shipinstruct)), max(length(comment)) from local_pnb2dw1.oerling_lineitem_30k_nz
  where partkey between 200000000 and 6000000000
  group by returnflag, linestatus;




set session prism.orc_compression_codec='NONE';

create table local_pnb2dw1.oerling_lineitem_3k_nz with(retention_days=14, oncall='presto', format='DWRF', bucketed_by=array['orderkey'], bucket_count=512) as
select 
    orderkey,          
    partkey,           
    suppkey,           
    linenumber,       
    quantity,          
    extendedprice ,     
    discount ,          
    tax ,               
    returnflag ,    
    linestatus ,    
    cast(shipdate as varchar) as shipdate,            
    cast(commitdate as varchar) as commitdate,          
    cast(receiptdate as varchar) as receiptdate,         
    shipinstruct , 
    shipmode ,     
    comment        
from tpch.sf3000.lineitem;





create table local_pnb2dw1.oerling_orders_3k_nz with(retention_days=14, oncall='presto', format='DWRF', bucketed_by=array['orderkey'], bucket_count=512) as
select 
 orderkey , custkey , orderstatus , totalprice , cast(orderdate as varchar) as orderdate  , orderpriority ,      clerk      , shippriority ,                      comment                      
from tpch.sf3000.orders;

create table local_pnb2dw1.oerling_customer_3k_nz with(retention_days=14, oncall='presto', format='DWRF') as
select custkey ,        name        ,            address            , nationkey ,      phone      , acctbal , mktsegment ,                          comment
 from tpch.sf3000.customer;
 



create table local_pnb2dw1.oerling_part_3k_nz with(retention_days=14, oncall='presto', format='DWRF', bucketed_by=array['partkey'], bucket_count=512) as  
select * 
 from tpch.sf3000.part;
 

create table local_pnb2dw1.oerling_supplier_3k_nz with(retention_days=14, oncall='presto', format='DWRF') as  
select * 
 from tpch.sf3000.supplier;
 

create table local_pnb2dw1.oerling_nation_3k_nz with(retention_days=14, oncall='presto', format='DWRF') as  
select * 
 from tpch.sf1.nation;

create table local_pnb2dw1.oerling_region_3k_nz with(retention_days=14, oncall='presto', format='DWRF') as  
select * 
 from tpch.sf1.region;


create table local_pnb2dw1.oerling_partsupp_3k_nz with(retention_days=14, oncall='presto', format='DWRF', bucketed_by=array['partkey'], bucket_count=512) as  
select * 
 from tpch.sf3000.partsupp;
 



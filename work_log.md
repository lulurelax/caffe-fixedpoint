# Work Log
## Step 1
Substitute `Dtype` using `sg14::fixed_point<int32_t, -20>`. And modify the fixed point class:
  * add constructor, which initialize the data from int, float, or double.
  * modify its multiplication and division, make them generate a fixed point number with the same format(same length, and same bits of decimal).

## Step 2
Modify Caffe math functions, to make support fixed-point arithmetic. After run the demo, I got the wrong result. It turns out that the problems is caused by overflow. Therefore, 32 bits of fixed point is not applicable. the largest intermediate number is larger than 10,000.

## Step 3
Using `sg14::fixed_point<int64_t, -25>`, and solve some data width problems(by modifying all division into multiplication), we got the correct result.
> "n02123045 tabby, tabby cat"   
0.2380 - "n02123159 tiger cat"   
0.1235 - "n02124075 Egyptian cat"   
0.1003 - "n02119022 red fox, Vulpes vulpes"   
0.0715 - "n02127052 lynx, catamount"   

which is same with the float version of Caffe.

# Extending classifier capabilities
Contributors: @TonyBagnall

## Motivation

The classifiers are being enhanced to handle unequal length series and 
multivariate series. We need a mechanism for checking classifier capabilities. 
We also need a working model for handling unequal length series

# Capabilities: data types this classifier can handle
    capabilities = {
        "multivariate": False,
        "unequal_length": False,
        "missing_values": False,
    }
We need to decide
1. How to model this (i.e. do we fully adopt the tags from scikit learn)
2. How and where to check capabilities and what to do if they are not fulfilled
3. Do we add capabilities (e.g. ability to take numpy input for equal length classification)
4. Resolve the separation between checking and transforming data

## Use cases

The user tries to build a classifier on data in an unhandled data format (e.g. unequal length).
It should raise an informative error. 

## Other advantages

It will allow us to better summarise classifier capabilities and incrementally extend them to handle more 
complex formats

## Implementation details

I have currently added default capability tags and not used sklearn tags yet, 
since I dont like the mechanism and the tags system is not yet fully adoptyed in sktime (I think)

The capabilities are currently checked in the sktime methods check_X and check_X_y.
This can continue, but the associated issue is that check_X and 
check_X_y also handle data convertions from numpy to pandas.

I would remove the coercion from checkX_y, and have it simply check the classifier capabilities against the input data.

The assumption should be that if passed a 2D numpy it is a univariate, equal length problem, and
a 3D numpy is a multivariate, equal length problem. Any "missing" are assumed by the algorithm to be missing 
rather than padding. 

If conversion from or to numpy is required, it should be directly called after the capability checks. 
The assumption here is that responsibility for internal formatting is devolved to the classifier. 
Some may work directly with pandas, some convert to numpy. The sole job of checkX_y should be to check that the data satisfies the capabilities.
Ideally, I would also like these capabilities stored with the data, so these checks can be done in constant time. 





import numpy as np

def returnmultiple():
    return (['a','b','c'],([(1,2),(2,3)],True))

def passdict(keys,values):
    adict=dict(zip(keys,values))
    return (list(adict.keys()),list(adict.values()))

def editlist(alist,anum,astr,nestedlist):
    alist.append(5)
    anum += 1
    astr2 = astr+' Python'

    nestedlist[0].append(5)
    return astr2
def mockList(lst): # an array with ten elements
    #lst= np.array(lst)
    for i in range(len(lst)):
        lst[i]+=i

    check = np.zeros((4,3))
    check = check+3
    check = check.flatten()
    #lst.append(180)
    lst[:]=check # pass by reference for labview
    return check

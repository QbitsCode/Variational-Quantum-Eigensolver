def processStates(state): # List of states
    def aux(st): # Longueur = puissance de 2
        s = list(st)
        if len(s) == 2:
            return {'0' : s[0], '1' : s[1]}
        else:
            a0 = aux(s[:len(s)//2])
            a1 = aux(s[len(s)//2:])
            r = {}
            for k in a0:
                r['0' + k] = a0[k]
            for k in a1:
                r['1' + k] = a1[k]
            return r
    r = []
    for i in range(len(state)):
        r.append(aux(state[i]))
    return r


def printOneState(d): # get a dict as per processStates output
    start = " "
    for k in d:
        im = d[k].imag
        re = d[k].real
        if abs(im) >= 0.001 or abs(re) >= 0.001:
            print("%s % .3f + % .3fj |%s>" % (start,re,im,k))
            if start[0] == " ":
                start = "+"


def printOneState(d): # get a dict as per processStates output
    start = " "
    for k in d:
        im = d[k].imag
        re = d[k].real
        if abs(im) >= 0.001 or abs(re) >= 0.001:
            print("%s % .3f + % .3fj |%s>" % (start,re,im,k))
            if start[0] == " ":
                start = "+"

def strOneState(d):
    r = ''
    start = " "
    for k in d:
        im = d[k].imag
        re = d[k].real
        if abs(im) >= 0.001 or abs(re) >= 0.001:
            r += "%s % .3f + % .3fj |%s>" % (start,re,im,k) +'\n'
            if start[0] == " ":
                start = "+"
    return r

def getDictFinalRes(result):
    return processStates([result['statevector']])[0]
            
def printFinalRes(result):
    printOneState(processStates([result['statevector']])[0])

def getStrFinalRes(result):

    return strOneState(processStates([result['statevector']])[0])
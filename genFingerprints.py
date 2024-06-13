import numpy as np
import math
import pandas as pd
import random
'''
This generates an input file to be passed through the calibration software.
file: string; the path to the input file. 
numRadialFingerprints: integer; the number of different radial fingerprints to generate
numBondFingerprints: integer; the number of different bond fingerprints to generate
radialRange: (float,float); the min and max of range [min,max) to randomly generate fingerprints 
bondRange: (float,float); the min and max of range [min,max) to randomly generate fingerprints
'''

def genFingerprints(file: str, numRadialFingerprints: int, numBondFingerprints: int, radialRange: tuple[float, float], bondRange: tuple[float,float]) -> None:
    # open input file for reading   
    lines = ""
    try:
        with open(file, "r") as f:
            pairs = []
            lines = f.readlines()
            f.close()
    except:
        print("Error: could not find "+file)
        return
    
    # turn file into a data frame for easier munipulation
    firstLine = lines.pop(0)
    for i in range(0,len(lines)-1,2):
        pairs.append([lines[i].strip(),lines[i+1].strip()])
    df = pd.DataFrame(pairs,columns=["variables","values"])

    # read atom types
    atomtypes = [a for a in df[df["variables"]=="atomtypes:"]["values"]]
    if len(atomtypes) != 1:
        print("Error reading atoms!")
        return
        
    # compute |o - n|
    # know there should only be one of each
    try:
        o = [i for i in df[df['variables'].str.contains("radialscreened_0:o:")]["values"]][0]
        n = [i for i in df[df['variables'].str.contains("radialscreened_0:n:")]["values"]][0]
        o = int(o)
        n = int(n)
        alphas = 0
        # if o < 0 and n >0 then |o-n| will be off by 1
        if  ((o < 0) and (n > 0)) or ((o > 0) and (n < 0)):
            alphas = abs(o-n)+1
        else:
            alphas = abs(o-n)
    except:
        print("Error: find o and n for radial screened!")
        return
        
    # make sure we can generate all requested fingerprints
    if numRadialFingerprints % alphas != 0:
        print("Error: Requested number of radial fingerprints is not a multiple of alpha!")
        return
        
    # now we are going to make copies of the fingerprints
    # first we create templates to add to the data frame
    pattern = r'^fingerprintconstants:'+atomtypes[0]+'_'+atomtypes[0]+':radialscreened_0:.*$'
    radialTemplate = df[df['variables'].str.contains(pattern, regex=True)]
    pattern = r'^fingerprintconstants:'+atomtypes[0]+'_'+atomtypes[0]+'_'+atomtypes[0]+':bondscreened_0:.*$'
    bondTemplate = df[df['variables'].str.contains(pattern, regex=True)]
    if radialTemplate.empty or bondTemplate.empty:
        print("Error: missing bond or radial screened 0 entries!")
        return 
    
    # remove all fingerprints
    df = df[~df['variables'].str.contains("radialscreened")]
    df = df[~df['variables'].str.contains("bondscreened")]
        
    # now we add on blocks of radial fingerprints
    radialBlocks  = int(numRadialFingerprints / alphas)
    radialFingerprints = ""
    for i in range(radialBlocks):
        # need to say how many blocks we are adding
        radialFingerprints += "radialscreened_"+str(i)+" "
        # generate new alphas
        radialI = radialTemplate.copy()
        newValues = ""
        lowerBound = radialRange[0]
        upperBound = radialRange[1]
        for _ in range(alphas):
            n = random.uniform(upperBound,lowerBound)
            newValues += str(n) + " "
            
        # change from radialscreened_0 to radialscreened_i
        radialI["variables"] = radialI["variables"].str.replace("_0:","_"+str(i)+":")
            
        # add new alpha values
        radialI.loc[radialI["variables"]=="fingerprintconstants:"+atomtypes[0]+"_"+atomtypes[0]+":radialscreened_"+str(i)+":alpha:","values"] = newValues
        # combine with existing input file
        df = pd.concat([df,radialI],ignore_index=True)
    
    # update with amount of radial fingerprints we added
    df.loc[df["variables"]=="fingerprints:"+atomtypes[0]+"_"+atomtypes[0]+":","values"] = radialFingerprints

    # now we create the new bond fingerprints
    # set k to be amount we create
    bondTemplate.loc[bondTemplate["variables"]=="fingerprintconstants:"+atomtypes[0]+'_'+atomtypes[0]+'_'+atomtypes[0]+":bondscreened_0:k:","values"] = numBondFingerprints

    # generate new alpha_k values
    newValues = ""
    lowerBound = bondRange[0]
    upperBound = bondRange[1]
    for _ in range(numBondFingerprints):
        n = random.uniform(upperBound,lowerBound)
        newValues += str(n) + " "
        
    # add alpha_k to bond template
    bondTemplate.loc[bondTemplate["variables"]=="fingerprintconstants:"+atomtypes[0]+'_'+atomtypes[0]+'_'+atomtypes[0]+":bondscreened_0:alphak:","values"] = newValues
    df  = pd.concat([df,bondTemplate],ignore_index=True)

    # update with total amount of fingerprints added
    df.loc[df["variables"]=="fingerprintsperelement:"+atomtypes[0]+":","values"] = str(radialBlocks+1)

    # we set the input layer size to m*k+#n's
    m = df.loc[df["variables"]=="fingerprintconstants:"+atomtypes[0]+'_'+atomtypes[0]+'_'+atomtypes[0]+":bondscreened_0:m:","values"]
    m = int(m.item())
    df.loc[df["variables"]=="layersize:"+atomtypes[0]+":0:","values"] = (m*numBondFingerprints)+alphas 

    # finally we write the data frame back to the input file
    df = df.to_numpy()
    with open(file, "w") as f:
        f.write(firstLine)
        for row in df:
            f.write(row[0]+"\n"+str(row[1])+"\n")
        f.close()
    print("Finished generating fingerprints!")

if __name__ == "__main__":
    genFingerprints("_Ti copy.nn",5,5,(80,100),(-5,100))

    

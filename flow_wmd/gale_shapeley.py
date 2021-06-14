import copy
import numpy as np
 
# Adapted from: https://rosettacode.org/wiki/Stable_marriage_problem#Python
    
class Matcher():
    def __init__(self, D:np.array):
        X1_pref = {str(idx): list(row.argsort()) for (idx, row) in enumerate(D)}
        X2_pref = {str(idx): list(row.argsort()) for (idx, row) in enumerate(D.T)}
        
        X1_pref = {kk: [str(v) for v in vv] for kk, vv in X1_pref.items()}
        X2_pref = {kk: [str(v) for v in vv] for kk, vv in X2_pref.items()}
        
        self.guyprefers = X1_pref
        self.galprefers = X2_pref

        self.guys = sorted(X1_pref.keys())
        self.gals = sorted(X2_pref.keys())

    def matchmaker(self) -> dict:
        guysfree = self.guys[:]
        engaged  = {}
        guyprefers2 = copy.deepcopy(self.guyprefers)
        galprefers2 = copy.deepcopy(self.galprefers)
        while guysfree:
            guy = guysfree.pop(0)
            guyslist = guyprefers2[guy]
            gal = guyslist.pop(0)
            fiance = engaged.get(gal)
            if not fiance:
                # She's free
                engaged[gal] = guy
                #print("  %s and %s" % (guy, gal))
            else:
                # The bounder proposes to an engaged lass!
                galslist = galprefers2[gal]
                if galslist.index(fiance) > galslist.index(guy):
                    # She prefers new guy
                    engaged[gal] = guy
                    #print("  %s dumped %s for %s" % (gal, fiance, guy))
                    if guyprefers2[fiance]:
                        # Ex has more girls to try
                        guysfree.append(fiance)
                else:
                    # She is faithful to old fiance
                    if guyslist:
                        # Look again
                        guysfree.append(guy)
        self.engaged = engaged
        engaged_int = {int(k): int(engaged[k]) for k in engaged.keys()}
        return engaged_int
        
    def check(self) -> bool:
        inverseengaged = dict((v,k) for k,v in self.engaged.items())
        for she, he in self.engaged.items():
            shelikes = self.galprefers[she]
            shelikesbetter = shelikes[:shelikes.index(he)]
            helikes = self.guyprefers[he]
            helikesbetter = helikes[:helikes.index(she)]
            for guy in shelikesbetter:
                guysgirl = inverseengaged[guy]
                guylikes = self.guyprefers[guy]
                if guylikes.index(guysgirl) > guylikes.index(she):
                    print("%s and %s like each other better than "
                          "their present partners: %s and %s, respectively"
                          % (she, guy, he, guysgirl))
                    return False
            for gal in helikesbetter:
                girlsguy = self.engaged[gal]
                gallikes = self.galprefers[gal]
                if gallikes.index(girlsguy) > gallikes.index(he):
                    print("%s and %s like each other better than "
                          "their present partners: %s and %s, respectively"
                          % (he, gal, she, girlsguy))
                    return False
        self.check = True
        return True
        
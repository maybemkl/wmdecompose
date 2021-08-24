import copy
import numpy as np

class Matcher():
    """
    Class for finding the optimal Gale-Shapeley matches from a distance matrix where rows are "guys" and columns are "gals", 
    following wording from the original GS paper.
    
    The algorithm is slightly biased to the "suitor", i.e. the "guy".
    
    Code adapted from: https://rosettacode.org/wiki/Stable_marriage_problem#Python
    """
    
    def __init__(self, D:np.array):
        """
        Initialize class instance by finding preferences for guys and gals. 
        The closer a "guys" is to "gal", the more preferable she is to him and vice versa.
        """
        
        guy_pref = {str(idx): list(row.argsort()) for (idx, row) in enumerate(D)}
        gal_pref = {str(idx): list(row.argsort()) for (idx, row) in enumerate(D.T)}
        
        guy_pref = {kk: [str(v) for v in vv] for kk, vv in guy_pref.items()}
        gal_pref = {kk: [str(v) for v in vv] for kk, vv in gal_pref.items()}
        
        self.guyprefers = guy_pref
        self.galprefers = gal_pref

        self.guys = sorted(guy_pref.keys())
        self.gals = sorted(gal_pref.keys())

    def matchmaker(self) -> dict:
        """
        Use the Gale Shapley algorithm to find a stable set of engagements
        """
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
        #engaged_int = {int(k): int(engaged[k]) for k in engaged.keys()}
        engaged_int = {int(engaged[k]): int(k) for k in engaged.keys()}
        return engaged_int
        
    def check(self) -> bool:
        """
        Perturb the set engagements given by matchmaker() to form an unstable set of engagements then check this new set for stability.
        """
        
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
                    print(f"{she} and {guy} like each other better than "
                          f"their present partners: {he} and {guysgirl}, respectively")
                    return False
            for gal in helikesbetter:
                girlsguy = self.engaged[gal]
                gallikes = self.galprefers[gal]
                if gallikes.index(girlsguy) > gallikes.index(he):
                    print(f"{he} and {gal} like each other better than "
                          f"their present partners: {she} and {girlsguy}, respectively")
                    return False
        self.check = True
        return True
        
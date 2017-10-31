from __future__ import print_function
import suds
import sys
import numpy as np
np.set_printoptions(precision=2)
import datetime
import dateutil.parser

class Pmtrs:
    """Parameters for using the modified PageRank to rank teams during a league"""
    class BL1:
        #parameters
        damping=.8      # Like the PageRank damping, TBD: should depend on the number of games
        halfLife=30     # days after which the results should influence the rating with 1/2
        goalDiff=.5    # to rate difference of goals 
        class weights:
            '''Weights used to rate the single games'''
            gv=0.55            # guest victory
            hv=0.4            # home victory
            gd=0.2            # draw as guest
            hd=0.0             # draw at home
        class forecast:
            '''Parameters used in forecast'''
            boundH=4     # H >= G - boundH 
            boundG=5     # G >= H - boundG    
            #width = boundH + boundG
			
class League:
    """Read data from openligadb.de and define basic properties"""
    _tooManyRequests = 40 # set to prevent too many requests to openligadb.de, set > expected # match days later
    def __init__(self, name, saison):
        ## e.g. name = "bl1", saison = "2017"
        self.name = name    # instance variable unique to each instance
        self.saison = saison
        
        ## Get Match-List
        URL = "http://www.openligadb.de/Webservices/Sportsdata.asmx?WSDL" # SOAP Interface
        client = suds.client.Client(URL) # Create client
      
        print("Downloading data...", end= ' ') #TBD: chache data!
        try:
            Matches = client.service.GetMatchdataByLeagueSaison(name, saison).Matchdata
        except AttributeError:
            print("n/A")
            sys.exit(1)
        print("done.")
        print(" ")
        
        ## Number of matches per day, as len(self.AllMatches) == nTeams*(nTeams-1)
        self.nMatchesPerDay = int( (np.sqrt(4*len(Matches)+1)+1)/4 )

        ## Total number of match days
        self.nMatchDays = 2*(2*self.nMatchesPerDay-1)
        
        self.AllMatches = []
        for j in range(self.nMatchDays):
            name = str(j+1)+". Spieltag"
            DatchDay = filter(lambda x: x.groupName  == name, Matches)
            self.AllMatches.append(DatchDay)
            if len(filter(lambda x: x.matchIsFinished, DatchDay)) == self.nMatchesPerDay:
                self.LastMatchday = j+1
                
        ## List of teams and TeamIds
        self.teams = [];
        for match in self.AllMatches[0]:
            #append [idTeam1,nameTeam1]
            self.teams.append((match.idTeam1,match.nameTeam1.encode("utf-8")))
            #append [idTeam2,nameTeam2]
            self.teams.append((match.idTeam2,match.nameTeam2.encode("utf-8")))
        
        ## Number of teams
        self.nTeams = len(self.teams)
    ## -------- end init ------------
        
    def printTeams(self):
        """Output teams and their IDs"""
        for team in self.teams:
            id, name = team
            print("{1:<23} {0:>4d}".format(id, name.encode("utf-8")))

    def printMatchDay(self,day):
        """Output last league day"""
        for match in self.AllMatches[day-1]:
            if match.matchResults:
                goals =  "[{0}:{1}]".format(
                    match.matchResults.matchResult[1].pointsTeam1, match.matchResults.matchResult[1].pointsTeam2)
            else:
                goals = "[N/A]"
            print( """{start_time} - {goals} - {team1} vs. {team2}""".format(
                team1=match.nameTeam1.encode("utf-8"),
                team2=match.nameTeam2.encode("utf-8"), 
                goals=goals, start_time=match.matchDateTime.strftime("%a, %H:%M Uhr")))
            
    def idTeam2index(self,teamId):
        indexAsArray = np.where(np.array(self.teams).T[0]==str(teamId))
        return indexAsArray[0][0]
        
    def idTeam2name(self,ID):
        return self.teams[self.idTeam2index(ID)][1]
		
		
		
class Rank:
    '''Ranking for the instance league of myLeague'''
    
    dayNoScore = 2        
    
    def __init__(self, league, pmtrs):
        '''input: instance league of myLeague, instance pmtrs of myPmtrs'''
        self.league = league
        self.pmtrs = pmtrs
    # --------- end init ------------
    
    def weightGoalDiff(self,n):
        x = self.pmtrs.goalDiff
        if n == 0:
            return 1
        if n == 1:
            return 1
        elif n==2:
            return 1 + x
        else:
            return (x+1)*self.weightGoalDiff(n-1)-x*self.weightGoalDiff(n-2)
        
    def _rateGame(self,typ,t,teamGoals,opponentGoals):
    #   type in [h,g],   result is t days in the past
        HalfLife = self.pmtrs.halfLife
        tau = HalfLife/np.log(2)
        decay = np.exp(-t/tau)
        diff = abs(teamGoals-opponentGoals)
        Tendenz = np.sign(teamGoals-opponentGoals)
    #    
        val = decay*self.weightGoalDiff(diff)
    #    
        if typ == 'h':
            if Tendenz == 1: #Heimsieg
                return self.pmtrs.weights.hv*val
            elif Tendenz == 0: #Heimunentschieden
                return self.pmtrs.weights.hd*val
            else: #Heimniederlage
                return self.pmtrs.weights.gv*val
        elif typ == 'g':
            if Tendenz == 1: #Auswaertssieg
                return self.pmtrs.weights.gv*val
            elif Tendenz == 0: #Auswaertsunentschieden
                return self.pmtrs.weights.gd*val
            else: #Auswaertsniederlage
                return self.pmtrs.weights.hv*val
        else:
            raise NameError('Wrong type')
    # ---- end rateGame ----
    
    def scoreMatrices(self,Day):
        '''output: tuple of matrices (AS,HS,AN,HN,S,N)'''
        Day = Day-1 # self.AllMatches starts counting at 0
        nVereine = self.league.nTeams
        HS,AS,HN,AN = [np.zeros([nVereine,nVereine]) for _ in range(4)]   
        #Generate matrix sieg[i,j]: RateGame( Vereine[i]:Vereine[j] ) 
        #
        for day in range(Day):
            st = self.league.AllMatches[day]
            for j in range(self.league.nMatchesPerDay):
                match = st[j]
                age = datetime.date.today() - match.matchDateTime.date()
                age = age.days
                def RG(typ,th,tg):
                    return self._rateGame(typ,age,th,tg)
                # Get indices of heim and gast team:    
                h = self.league.idTeam2index(match.idTeam1) # home
                g = self.league.idTeam2index(match.idTeam2) # guest
                toreHeim = match.matchResults.matchResult[1].pointsTeam1
                toreGast = match.matchResults.matchResult[1].pointsTeam2
                # execute ratings
                if toreHeim > toreGast:
                    HS[h,g] = RG('h', toreHeim, toreGast)
                    AN[g,h] = RG('g', toreGast, toreHeim)
                elif toreHeim < toreGast:
                    HN[h,g] = RG('h', toreHeim, toreGast)
                    AS[g,h] = RG('g', toreGast, toreHeim)
                elif toreHeim == toreGast:
                    HN[h,g] = RG('h', toreHeim, toreGast)
                    HS[h,g] = RG('h', toreHeim, toreGast)
                    AS[g,h] = RG('g', toreGast, toreHeim)
                    AN[g,h] = RG('g', toreGast, toreHeim)
                #end if
            #end for
        #end for
        return {"AS":AS, 
                "HS": HS, 
                "AN": AN, 
                "HN": HN,
                "S": HS + AS,
                "N": HN + AN}
    # -------- end scoreMatrices ---------
    
    def getRanking(self,day,ListNameScoreMat):
        '''Dictionary of rankings, always non-negative (for large enough damping)'''
        def normalize(list):
            '''normalize list by mean'''
            return 100*list/abs(list.mean())
        
        scoreMatDict = self.scoreMatrices(day)
        
        dictRank = {}
        dictNegVals = {}
        for name in ListNameScoreMat:
            M=scoreMatDict[name]
            
            # Calculate ranking matrix
            d = self.pmtrs.damping
            n = self.league.nTeams
            I = np.identity(n)
            constVector = (1-d)/n*np.ones(n)
            lam = 100/(1-d)
            dictRank[name] = normalize( lam*np.linalg.solve(I - d*M, constVector) )
            
            # Check for negative values
            dictNegVals[name] = sum(dictRank[name].reshape(-1)<0)
            if dictNegVals[name] > 0:
                print( "Nr. negative values in ranking matrix < {0} >: {1}".format(name, dictNegVals[name]) )

        return (dictRank, dictNegVals)
    # --------- end ScoreMat2RankMat ----------
  
    # -------------- output ------------------   
    ## TBD: order lists
    def printList(self,a,b,c):
        '''a: list victory ranking, b: list defeat ranking, c: list total ranking'''
        print('{:^27}    {}   {}   {}'.format(*('Verein','SRank','NRank','Rank')) )
        print('----------------------------------------------------')
        for j in range(self.league.nTeams):
            print('{:<27}{:8.2f}{:8.2f}{:8.2f}'.format(
                self.league.teams[j][1], a[j], b[j], c[j]))
    
    def _printCaption(self,string):
        n = 50;
        m = int((n-4.5-len(string))/2)
        print(n*'=')
        print(m*'=' + '   ' + string + '   ' + m*'=')
        print(n*'=')        
    
    def printTotal(self, day):
        dictRank, _ = self.getRanking(day,['S','N'])
        self._printCaption("Ranking")
        S = dictRank['S']
        N = dictRank['N']
        self.printList(S,N,S-N)
        
    def printHome(self,day):
        dictRank, _ = self.getRanking(day,['HS','HN'])
        self._printCaption("Heim ranking")
        HS = dictRank['HS']
        HN = dictRank['HN']
        self.printList(HS, HN, HS-HN)

    def printAway(self,day):
        dictRank, _ = self.getRanking(day,['AS','AN'])
        self._printCaption("Auswaerts ranking")
        AS = dictRank['AS']
        AN = dictRank['AN']
        self.printList(AS, AN, AS-AN)
        
    def printComparisonHG(self,day):
        '''Print list with ranking from list H vs. ranking from list A to given match day'''  
        dictR, _ = self.getRanking(day,['AS','AN','HS','HN'])
        H = dictR['HS'] - dictR['HN']
        A = dictR['AS'] - dictR['AN']
        #caption
        self._printCaption('HeimRank vs. Auswaertsrank')
        print( '{:<27} {:<23}  {}    {}'.format(*('Heim','Gast','HRank(Heim)','ARank(Gast)')) )
        #output
        for match in self.league.AllMatches[day-1]:
            iH = self.league.idTeam2index(match.idTeam1) #index of home team
            iG = self.league.idTeam2index(match.idTeam2) #index of guest team
            print('{:<27}{:<27}{:8.2f}     {:8.2f}'.format(
                self.league.teams[iH][1],
                self.league.teams[iG][1],
                H[iH],
                A[iG] ))
            
    def printComparison(self,day):
        '''Print list with total ranking coresponding to given match day'''  
        dictR, _ = self.getRanking(day-1,['S','N'])
        totRank = dictR['S'] - dictR['N']
        #caption
        self._printCaption('Total Ranking')
        print( '{:<27} {:<23}  {}    {}'.format(*('Heim','Gast','Rank(Heim)','Rank(Gast)')) )
        #output
        for match in self.league.AllMatches[day-1]:
            iH = self.league.idTeam2index(match.idTeam1) #index of home team
            iG = self.league.idTeam2index(match.idTeam2) #index of guest team
            print('{:<27}{:<27}{:8.2f}     {:8.2f}'.format(
                self.league.teams[iH][1],
                self.league.teams[iG][1],
                totRank[iH],
                totRank[iG] ))
        
    # predict next match day
    def getPrediction(self,day):
        '''Predict matchday <day> based on scoreMatrices <day-1>'''
        def normalize(list):
            OneNorm = sum(map(abs,list))
            return 100*list/OneNorm
        def predictTendency(self,hR,gR):
            '''Predict tendency from hR (number for heim ranking score) and gR (guest ranking score)'''
            if hR-gR >= self.pmtrs.forecast.boundH:
                return 'H' #home team wins
            elif -self.pmtrs.forecast.boundG >= hR-gR:
                return 'G' #guest teams wins
            else:
                return 'D' #draw
        dictR, _ = self.getRanking(day-1,['AS','AN','HS','HN'])
        tp = [] # tendendy prediction list
        for match in self.league.AllMatches[day-1]:
            iH = self.league.idTeam2index(match.idTeam1)
            iG = self.league.idTeam2index(match.idTeam2)
            rankH = normalize(dictR['HS'] - dictR['HN'])
            rankA = normalize(dictR['AS'] - dictR['AN'])
            hR = rankH[iH]
            gR = rankA[iG]
            tp.append([day,
                       match.idTeam1, match.nameTeam1, hR,
                       match.idTeam2, match.nameTeam2, gR,
                       predictTendency(self,hR,gR) ])
        return tp   
    
    def printPrediction(self,day):
        for x in self.getPrediction(day):
            print( "{}  {:<24}  (score: {:<4.2f}) -   {:<24} (score: {:<4.2f})    {:>2}".format(
                  x[0], x[2].encode("utf-8"), x[3] , x[5].encode("utf-8"), x[6], x[7]) )
    
    def getTendencies(self,day):
        tendencies = []
        for match in self.league.AllMatches[day-1]:
            iH = self.league.idTeam2index(match.idTeam1)
            iG = self.league.idTeam2index(match.idTeam2)
            goalsTeam1 = match.matchResults.matchResult[1].pointsTeam1
            goalsTeam2 = match.matchResults.matchResult[1].pointsTeam2
            if goalsTeam1 > goalsTeam2: #home win
                tendency = 'H' 
            elif goalsTeam1 == goalsTeam2: #draw 
                tendency = 'D'
            else: #guest win
                tendency = 'G'
            tendencies.append([day,
                               match.idTeam1, match.nameTeam1,
                               match.idTeam2, match.nameTeam2,
                               tendency ])
        return tendencies
        
    def printTendencies(self,day):
        for x in self.getTendencies(day):
            print( "{}  {:<24} - {:<24}   {:>2}".format(
                  x[0], x[2].encode("utf-8"), x[4].encode("utf-8"), x[5]) )
        
    # evaluate rating system
    def score(self, day):
        '''Return score of comparing prediction of matchday <day> with results of <day>'''
        correct = 1 # right tendency
        wrong = 0 # wrong tendency, one of them draw
        verywrong = 0 # wrong tendendy, {prediction, result}={home win, home defeat}

        def evalScore(t1,t2):
            if t1==t2:
                return correct
            elif set([t1,t2]) == set(['H','G']):
                return verywrong
            else:
                return wrong
        
        actualTendencies = self.getTendencies(day)
        predictedTendencies = self.getPrediction(day)
        s = 0
        aDraw = 0
        pDraw = 0
        for j in range(self.league.nMatchesPerDay):
            pT = predictedTendencies[j][-1]
            aT = actualTendencies[j][-1]
            if pT == aT:
                s = s+1
            if pT == 'D':
                pDraw = pDraw+1
            if aT == 'D':
                aDraw = aDraw+1
        return (s, pDraw, aDraw)
    
    def accumulatedScore(self,day):
        '''Returns total score until <= min(day, league.LastMatchDay-1)'''
        #TBD: also count home wins and home defeats
        Day = min( day, self.league.LastMatchday - 1 )
        score = 0
        aDraw = 0
        pDraw = 0
        #ignore first days <= self.dayNoScore days in rating
        for d in range(self.dayNoScore,Day): #range(a,b) = a,...,b-1
            (s,pD,aD) = self.score(d+1)
            score = score + s
            aDraw = aDraw + aD
            pDraw = pDraw + pD
        return (score, pDraw, aDraw)
    
    def printAccumulatedScore(self,day):
        score, pDraw, aDraw = self.accumulatedScore(day)
        Day = min( day, self.league.LastMatchday - 1 )
        maxScore = (Day-self.dayNoScore)*self.league.nMatchesPerDay
        print("Score: {} out of {}".format(score,maxScore))
        print("Total of predicted draws: {}, total actual draws: {}".format(pDraw,aDraw))
        
    def getMaxScore(self,day):
        Day = min( day, self.league.LastMatchday - 1 )
        return (Day-self.dayNoScore)*self.league.nMatchesPerDay
        
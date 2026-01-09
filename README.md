# thesis
Code for my master's thesis.

Open tasks:
- [x] start training from main
- [ ] adapt feature extractors (think about which features i want)
- [x] mine rules
- [ ] update process
    - [x] update method for rules to exlude specific states
    - [x] check whether less violations are possible in asp
    - [x] what to do if backtracking exceeds horizon? (should this even happen? No)
    - [ ] fix "looping" in same violation for a few steps that happens from time to time
    - [x] store updated rule set
    - [x] everything can be enumerated because of the intervals -> adapt rule update process
    - [ ] save rules as string in pickle instead of object (-> add some tag to identify udpated rules) 
- [ ] documentation/comments
- [ ] add more complex norms
- [x] evaluation: 
  - [x] count violations
  - [x] count number of used updated rules
- [ ] license(s)
- [ ] attributions for legible, oftendeeprl
- [x] (re)check (categorical) features in rule mining
- [x] clean up episode data collection
- [x] (don't) consider (absolute) features e.g. x, y in some rules?
- [ ] rule pruning (check retention):
  - [ ] every time after blocking all actions, since this adds a lot of rules?
  - [ ] check if duplicates even can occur, if yes eliminate 
  - [ ] instead of adding and then removing rules, check retention first
- [ ] use random subset of features for adapting rules?
- [x] rule tag for creation
- [x] currently rules are updated at the wrong state
- [x] check how intervals affect mined rules



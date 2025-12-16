class RuleSnapshot:
    enforceable_rules = dict()
    cancelable_rules = dict()
    rule_indices = []

    def __init__(self, enforceable_rules, cancelable_rules):
        self.enforceable_rules = enforceable_rules 
        self.cancelable_rules = cancelable_rules  
        self.rule_indices = range(0, len(list(enforceable_rules.keys()) + list(cancelable_rules.keys())))
from CTAFlow.data import DataClient, CrossProductEngine, CrossSpreadLeg

rb_leg = CrossSpreadLeg.load_from_dclient("RB", base_weight=2.0, start_date="2020-01-1")
ho_leg = CrossSpreadLeg.load_from_dclient("HO", base_weight=1.0, start_date="2020-01-01")
cl_leg = CrossSpreadLeg.load_from_dclient("CL", base_weight=-3.0, start_date="2020-01-01")

crack_spread = CrossProductEngine.from_legs([cl_leg, ho_leg, rb_leg])
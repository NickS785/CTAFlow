from ..data.raw_formatting.spread_manager import SpreadData

cl_f = SpreadData("CL")
static_df = cl_f.to_df("static")
dynamically_rolled = cl_f.to_df('sequential')

static_df.tail(25).to_csv('F:\\Downloads\\static_df.csv')
dynamically_rolled.tail(25).to_csv("F:\\Downloads\\seq_df.csv")



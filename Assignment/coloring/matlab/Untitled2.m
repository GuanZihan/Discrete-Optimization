clear
sdpvar x
optimize(x>=0,x,sdpsettings('solver',solver, 'verbose',1))
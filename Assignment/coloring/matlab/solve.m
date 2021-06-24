function [result] = solve(edges, vertex_num)
    color_num = 3;
    language = "yalmip";
    solver = "gurobi";
    vertex_num = double(vertex_num);
    X = binvar(vertex_num, color_num);
    W = binvar(color_num, 1);
    constraints = [];
    for i = 1:size(edges)
        for j = 1: color_num
%             constraints = [constraints, X(edges{i}{1} + 1, j) + X(edges{i}{2} + 1, j) <= W(j)];
        end
    end
    for i = 1:vertex_num
        exp = 0;
        for j = 1: color_num
            exp = exp + X(i, j);
        end
        constraints = [constraints, exp == 1];
    end
    obj = 0;
    for i = 1: color_num
        obj = obj + W(i);
    end
    disp(obj);
    out = optimize(constraints, obj,sdpsettings('solver',solver, 'verbose',1));
    disp(out);
    result = [];
end
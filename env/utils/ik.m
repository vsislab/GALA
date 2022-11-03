syms w1 w2 w3 w4 
syms a b c x y z
T1 = [-sin(w1) -cos(w1) 0 0;0 0 1 0;-cos(w1) sin(w1) 0 0;0 0 0 1];
T2 = [cos(w2) -sin(w2) 0 0;0 0 -1 -a;sin(w2) cos(w2) 0 0;0 0 0 1];
T3 = [cos(w3) -sin(w3) 0 b;sin(w3) cos(w3) 0 0 ;0 0 1 0;0 0 0 1];
T4 = [cos(w4) -sin(w4) 0 c;sin(w4) cos(w4) 0 0 ;0 0 1 0;0 0 0 1];
T1*T2*T3*T4;
T=[0 0 0 x;0 0 0 y;0 0 0 z;0 0 0 1];
ans1=simplify(inv(T2)*inv(T1)*T)
ans2=simplify(T3*T4)
simplify((y*sin(w2) - z*cos(w1)*cos(w2) - x*cos(w2)*sin(w1))^2 + (y*cos(w2) + z*cos(w1)*sin(w2) + x*sin(w1)*sin(w2))^2);
simplify((y*sin(w2) - z*cos(w1)*cos(w2) - x*cos(w2)*sin(w1))*cos(w2) - (y*cos(w2) + z*cos(w1)*sin(w2) + x*sin(w1)*sin(w2))*sin(w2));
%https://blog.csdn.net/l1216766050/article/details/96899701
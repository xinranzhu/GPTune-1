close all
clear all
clc






% obj_single=[[ 0.9536]
%  [ 0.5793]
%  [ 0.3569]
%  [ 2.3909]
%  [36.1963]
%  [ 0.5688]
%  [ 3.7281]
%  [19.3605]
%  [ 2.2345]
%  [ 2.9628]
%  [ 0.767 ]
%  [ 0.707 ]
%  [ 2.5673]
%  [ 6.7035]
%  [ 1.2919]
%  [ 0.5567]
%  [ 3.0686]
%  [ 2.1897]
%  [ 3.1739]
%  [ 0.6422]
%  [ 1.6189]
%  [ 1.6965]
%  [ 4.0461]
%  [ 0.3791]
%  [ 3.3478]
%  [11.8807]
%  [29.9983]
%  [ 2.4509]
%  [ 0.8304]
%  [ 3.9637]
%  [ 0.9479]
%  [ 0.9185]
%  [ 1.59  ]
%  [ 1.1251]
%  [ 0.8565]
%  [ 0.8765]
%  [ 3.1207]
%  [ 1.9026]
%  [ 0.4636]
%  [11.3191]
%  [ 0.8154]
%  [ 1.4796]
%  [ 1.6032]
%  [ 0.4568]
%  [ 1.8519]
%  [34.2359]
%  [ 3.8902]
%  [17.7841]
%  [ 2.456 ]
%  [ 0.5341]
%  [ 0.6783]
%  [ 0.6913]
%  [ 2.419 ]
%  [ 8.8882]
%  [ 1.2103]
%  [ 0.602 ]
%  [ 2.706 ]
%  [ 2.2873]
%  [ 3.0425]
%  [ 0.5608]
%  [ 0.9215]
%  [ 1.4168]
%  [ 5.2624]
%  [ 0.5929]
%  [ 4.1534]
%  [10.9491]
%  [29.2093]
%  [ 2.2904]
%  [ 0.5948]
%  [ 4.0447]
%  [ 0.7244]
%  [ 1.0074]
%  [ 1.8587]
%  [ 1.2106]
%  [ 5.928 ]
%  [ 1.2162]
%  [ 3.9405]
%  [ 2.0431]
%  [ 0.6169]
%  [13.0817]
%  [ 0.8925]
%  [ 1.6265]
%  [ 1.2898]
%  [ 0.5123]
%  [ 1.8453]
%  [36.0239]
%  [ 3.7346]
%  [17.7228]
%  [ 1.9432]
%  [ 0.5372]
%  [ 0.759 ]
%  [ 0.7313]
%  [ 5.1577]
%  [ 8.2286]
%  [ 1.1392]
%  [ 0.4658]
%  [ 4.1705]
%  [ 2.0852]
%  [ 2.7439]
%  [ 0.4941]
%  [ 0.7962]
%  [ 1.1652]
%  [ 5.3414]
%  [ 0.6255]
%  [ 3.7692]
%  [10.963 ]
%  [29.3219]
%  [ 2.1395]
%  [ 1.2615]
%  [ 3.6273]
%  [ 0.7566]
%  [ 0.9087]
%  [ 1.6887]
%  [ 3.216 ]
%  [ 2.7691]
%  [ 2.7062]
%  [ 0.9566]
%  [ 3.5767]
%  [ 1.6016]
%  [ 0.8017]
%  [12.9732]
%  [ 0.6361]
%  [ 2.6365]
%  [ 0.6588]
%  [ 0.4063]
%  [ 1.7304]
%  [37.6945]
%  [ 3.9121]
%  [17.6546]
%  [ 1.7904]
%  [ 0.6936]
%  [ 3.9768]
%  [ 0.564 ]
%  [ 2.7518]
%  [ 7.5841]
%  [ 1.0476]
%  [ 0.5012]
%  [ 4.7516]
%  [ 2.8356]
%  [ 2.8523]
%  [ 0.4957]
%  [ 1.1378]
%  [ 1.2212]
%  [ 4.4467]
%  [ 4.0711]
%  [ 3.8656]
%  [11.008 ]
%  [29.0754]
%  [ 2.1196]
%  [ 0.7332]
%  [ 3.0246]
%  [ 0.7318]
%  [ 0.8937]
%  [ 1.8914]
%  [ 0.8693]
%  [ 0.8128]
%  [ 0.7452]
%  [ 3.5891]
%  [ 1.6139]
%  [ 0.6016]
%  [12.351 ]
%  [ 0.8294]
%  [ 1.2464]
%  [ 0.7156]
%  [ 0.4483]
%  [ 1.962 ]
%  [37.081 ]
%  [ 3.7697]
%  [17.5109]
%  [ 1.9733]
%  [ 0.434 ]
%  [ 0.7318]
%  [ 0.6009]
%  [ 2.5332]
%  [ 7.7499]
%  [ 1.052 ]
%  [ 0.4939]
%  [ 2.7234]
%  [ 2.3606]
%  [ 2.7527]
%  [ 0.6802]
%  [ 0.8345]
%  [ 1.1832]
%  [ 3.9427]
%  [ 0.4611]
%  [ 3.2421]
%  [10.7293]
%  [ 2.0663]
%  [ 0.5925]
%  [ 3.3339]
%  [ 0.838 ]
%  [ 0.957 ]
%  [ 1.7608]
%  [ 0.9736]
%  [ 1.0958]
%  [ 2.786 ]
%  [ 9.1385]
%  [ 5.6634]
%  [ 0.434 ]
%  [10.435 ]
%  [ 1.5827]
%  [16.5842]
%  [21.9199]
%  [ 1.4152]
%  [38.7402]
%  [ 0.5747]
%  [ 0.6842]
%  [ 0.73  ]
%  [ 6.1273]
%  [ 2.3043]
%  [ 6.2565]
%  [ 0.5304]
%  [19.6703]
%  [ 0.548 ]
%  [ 0.5904]
%  [ 5.2909]
%  [50.475 ]
%  [47.326 ]
%  [ 2.3702]
%  [ 0.2257]
%  [ 0.9133]
%  [ 2.0782]
%  [ 0.3919]
%  [ 4.6457]
%  [ 2.1287]
%  [ 0.4459]
%  [ 2.7648]
%  [14.7253]
%  [ 0.4479]
%  [ 1.0765]
%  [12.8062]
%  [13.4514]
%  [10.1566]
%  [ 1.4633]
%  [ 0.4361]
%  [38.01  ]
%  [12.533 ]
%  [47.5425]
%  [ 3.6191]
%  [ 2.1973]
%  [18.7074]
%  [ 0.5682]
%  [ 1.0703]
%  [ 8.6316]
%  [18.2742]
%  [18.0081]
%  [ 2.1923]
%  [ 6.7676]
%  [ 6.3025]
%  [ 3.6122]
%  [45.5168]
%  [38.5686]
%  [ 0.7271]
%  [ 0.5357]
%  [ 3.451 ]
%  [17.012 ]
%  [ 1.4428]
%  [ 9.3833]
%  [ 1.102 ]
%  [ 1.1479]
%  [ 0.381 ]
%  [10.7973]
%  [ 8.85  ]
%  [ 2.08  ]
%  [ 0.2412]
%  [ 6.1163]
%  [14.2563]
%  [ 0.6099]
%  [ 4.713 ]
%  [ 7.8635]
%  [42.8696]
%  [ 1.1495]
%  [ 8.2552]
%  [ 2.3385]
%  [ 1.1503]
%  [ 1.7452]
%  [ 1.0601]
%  [ 4.7224]
%  [ 0.622 ]
%  [ 0.7017]
%  [ 1.9184]
%  [18.4454]
%  [49.7188]
%  [ 0.3841]
%  [ 0.3921]
%  [10.2355]
%  [ 0.3435]
%  [26.6308]
%  [ 1.4938]
%  [ 2.6298]
%  [ 0.4218]
%  [ 0.5844]
%  [14.5061]
%  [ 0.9362]
%  [ 0.4568]
%  [ 2.0055]
%  [ 0.5165]
%  [ 1.7976]
%  [ 1.9149]
%  [ 1.1429]
%  [ 0.3778]
%  [ 0.7668]
%  [ 0.7786]
%  [ 0.4697]
%  [ 0.7508]
%  [ 0.3097]
%  [ 2.9299]
%  [ 0.5177]
%  [ 0.5023]
%  [ 0.3814]
%  [ 0.9973]
%  [ 0.395 ]
%  [15.7802]
%  [16.4696]
%  [ 1.0273]
%  [ 2.3039]
%  [ 2.232 ]
%  [ 1.2107]
%  [ 0.4989]
%  [ 1.2923]
%  [ 0.3881]
%  [ 3.4181]
%  [37.3605]
%  [ 1.5328]
%  [ 0.739 ]
%  [ 3.5443]
%  [11.4166]
%  [39.0233]
%  [ 0.3026]
%  [ 3.4179]
%  [45.7851]
%  [ 0.4461]
%  [30.4396]
%  [ 2.7544]
%  [ 0.964 ]
%  [43.5592]
%  [ 1.5951]
%  [ 3.5543]
%  [ 0.9782]
%  [ 1.7353]
%  [ 0.7907]
%  [ 1.6355]
%  [ 0.3509]
%  [ 2.0464]
%  [ 1.0109]
%  [28.9251]
%  [ 0.2985]
%  [ 0.449 ]
%  [ 0.5411]
%  [ 1.0966]
%  [ 2.8353]
%  [ 1.2442]
%  [ 1.578 ]
%  [ 2.3422]
%  [ 0.4789]
%  [ 0.4967]
%  [ 0.4567]
%  [ 0.6145]
%  [ 1.2854]
%  [ 1.1986]
%  [32.5845]
%  [ 1.844 ]
%  [ 0.9632]
%  [ 1.0308]
%  [ 0.3912]
%  [ 0.5206]
%  [ 2.3696]
%  [ 0.3855]
%  [ 2.2299]
%  [ 2.4791]
%  [ 1.4994]
%  [ 2.3608]
%  [24.2955]
%  [ 0.4441]
%  [ 1.1012]
%  [ 0.471 ]
%  [ 1.8078]
%  [ 0.9455]
%  [ 2.065 ]
%  [ 3.3855]
%  [ 0.5049]
%  [ 0.571 ]
%  [ 1.9329]
%  [ 1.9932]
%  [ 3.6463]
%  [ 1.4825]
%  [ 0.8984]
%  [ 2.629 ]
%  [ 2.8959]
%  [ 0.4495]
%  [ 0.8266]
%  [ 0.8918]
%  [ 3.6436]
%  [ 0.5012]
%  [ 3.0335]
%  [ 2.0324]
%  [ 0.339 ]
%  [ 0.4355]
%  [ 1.2897]
%  [ 1.9574]];

obj_single= [[ 2.4394]
 [ 1.0062]
 [ 0.8912]
 [43.6559]
 [ 2.1144]
 [ 4.4687]
 [21.6779]
 [ 4.2005]
 [ 2.5459]
 [ 2.4446]
 [ 6.8072]
 [ 8.869 ]
 [ 2.8876]
 [ 1.7552]
 [ 4.094 ]
 [ 0.9609]
 [ 1.6945]
 [ 2.5832]
 [ 4.1297]
 [ 1.3378]
 [ 5.1682]
 [ 7.9093]
 [27.9253]
 [ 3.8774]
 [ 1.2434]
 [ 1.5808]
 [ 3.063 ]
 [ 1.8905]
 [ 1.9497]
 [ 1.8536]
 [ 1.3776]
 [14.0308]
 [ 1.3578]
 [ 2.1476]
 [ 1.2286]
 [ 1.1113]
 [39.8732]
 [ 4.1914]
 [20.5877]
 [ 2.737 ]
 [ 0.7749]
 [ 2.0978]
 [ 3.0934]
 [ 8.999 ]
 [ 2.4286]
 [ 2.7037]
 [ 3.8909]
 [ 0.9751]
 [ 2.1888]
 [ 1.9135]
 [ 2.6842]
 [ 0.9419]
 [ 4.0665]
 [ 8.0283]
 [29.1045]
 [ 3.4217]
 [ 1.0303]
 [ 1.7411]
 [ 3.6598]
 [ 1.6604]
 [ 1.6279]
 [ 1.5137]
 [ 0.978 ]
 [14.5307]
 [ 1.8221]
 [ 2.2332]
 [ 1.5533]
 [ 1.0079]
 [37.8112]
 [ 3.72  ]
 [21.778 ]
 [ 3.1245]
 [ 0.8494]
 [ 1.108 ]
 [ 3.6239]
 [ 7.7396]
 [ 1.8792]
 [ 0.9182]
 [ 3.8976]
 [ 0.7831]
 [ 1.4509]
 [ 2.4547]
 [ 4.4166]
 [ 1.0186]
 [ 3.6775]
 [ 8.7175]
 [29.6054]
 [ 3.2968]
 [ 1.1459]
 [ 1.3485]
 [ 3.148 ]
 [ 1.9997]
 [ 1.4409]
 [ 1.7227]
 [ 1.1384]
 [13.7743]
 [ 1.4166]
 [ 1.8316]
 [ 1.3864]
 [ 0.7406]
 [37.9009]
 [ 3.5737]
 [21.7487]
 [ 2.618 ]
 [ 1.1875]
 [ 1.2717]
 [ 3.9966]
 [ 7.2993]
 [ 1.9553]
 [ 0.9663]
 [ 3.82  ]
 [ 0.7039]
 [ 1.8779]
 [ 1.7752]
 [ 0.901 ]
 [ 3.6914]
 [10.1841]
 [29.8774]
 [ 3.3934]
 [ 1.0136]
 [ 1.4423]
 [ 2.3918]
 [ 1.5919]
 [ 1.1683]
 [ 1.5345]
 [ 0.81  ]
 [11.5281]
 [ 1.3116]
 [ 2.0035]
 [ 1.1107]
 [ 0.8397]
 [39.961 ]
 [ 4.1069]
 [20.6047]
 [ 2.711 ]
 [ 0.9654]
 [ 1.187 ]
 [ 3.5176]
 [ 8.5998]
 [ 1.3057]
 [ 0.7517]
 [ 2.6842]
 [ 1.0553]
 [ 1.2383]
 [ 1.749 ]
 [ 2.7691]
 [ 0.6355]
 [ 3.8915]
 [ 8.4467]
 [ 3.3941]
 [ 0.8579]
 [ 1.5568]
 [ 2.3781]
 [ 1.6536]
 [ 1.2636]
 [ 1.5307]
 [ 0.8336]
 [11.5909]
 [ 0.9811]
 [ 1.7259]
 [ 1.2721]
 [ 0.8064]
 [32.7612]
 [ 3.6934]
 [21.7186]
 [ 2.5064]
 [ 1.0064]
 [ 1.0483]
 [ 3.936 ]
 [ 7.4255]
 [ 1.8166]
 [ 0.9378]
 [ 0.9736]
 [ 1.1023]
 [ 2.3839]
 [ 2.0549]
 [ 0.9411]
 [ 3.411 ]
 [ 9.6224]
 [29.976 ]
 [ 3.1767]
 [ 1.0012]
 [ 1.4651]
 [ 2.3568]
 [ 1.6022]
 [ 1.2519]
 [ 1.8506]
 [ 0.7893]
 [12.6396]
 [ 1.0833]
 [ 2.2987]
 [ 1.2627]
 [ 0.8371]
 [40.8874]
 [ 0.7905]
 [ 3.708 ]
 [21.5482]
 [ 2.7374]
 [ 0.8248]
 [ 1.287 ]
 [ 1.1961]
 [ 4.8263]
 [17.7592]
 [ 2.924 ]
 [ 1.4001]
 [ 1.2849]
 [ 2.4997]
 [ 0.4259]
 [ 0.8745]
 [ 0.8166]
 [ 3.659 ]
 [ 1.1575]
 [50.1182]
 [ 4.6975]
 [ 4.9256]
 [ 4.7723]
 [ 1.4965]
 [ 2.2767]
 [27.8236]
 [ 0.8716]
 [ 4.8444]
 [ 0.7985]
 [ 1.4108]
 [ 1.1174]
 [ 1.0909]
 [ 4.6854]
 [ 2.1431]
 [50.0293]
 [20.7656]
 [ 3.8562]
 [38.5433]
 [ 2.0568]
 [36.9921]
 [ 0.3377]
 [38.4403]
 [14.9797]
 [29.3979]
 [27.5976]
 [34.4599]
 [ 1.47  ]
 [ 0.594 ]
 [ 2.1074]
 [ 0.8655]
 [50.4124]
 [ 1.0126]
 [ 2.6177]
 [ 5.1491]
 [ 2.168 ]
 [ 0.7392]
 [ 1.1057]
 [ 5.0599]
 [ 1.5286]
 [ 2.2576]
 [ 1.6931]
 [ 1.2163]
 [ 4.3136]
 [ 2.1059]
 [28.4767]
 [ 1.4703]
 [45.252 ]
 [41.8233]
 [ 1.6676]
 [ 2.6411]
 [ 2.1037]
 [ 4.8422]
 [ 1.2289]
 [ 1.5696]
 [20.5205]
 [29.1728]
 [29.4077]
 [ 1.3433]
 [ 0.7529]
 [ 1.0515]
 [52.6866]
 [26.674 ]
 [17.7549]
 [ 1.0855]
 [ 1.3668]
 [ 0.606 ]
 [ 0.767 ]
 [34.4439]
 [ 1.4908]
 [ 0.6976]
 [ 0.984 ]
 [28.6683]
 [ 0.6407]
 [ 4.925 ]
 [ 3.9446]
 [ 4.2542]
 [ 3.5977]
 [35.3072]
 [ 2.0337]
 [15.0811]
 [ 1.2968]
 [42.7383]
 [ 1.0736]
 [ 1.808 ]
 [ 1.1346]
 [44.8029]
 [ 1.9428]
 [ 1.8572]
 [ 1.7193]
 [ 0.8718]
 [ 1.7252]
 [16.2702]
 [ 5.4808]
 [ 0.3554]
 [49.5924]
 [ 1.4946]
 [ 5.9677]
 [ 4.1632]
 [ 0.7678]
 [ 1.3397]
 [ 1.5135]
 [ 1.2632]
 [ 2.0143]
 [50.4957]
 [27.3706]
 [ 1.1511]
 [ 4.1998]
 [ 0.6716]
 [ 2.0493]
 [ 1.1704]
 [ 1.2471]
 [ 0.5759]
 [ 1.0154]
 [ 1.9774]
 [ 2.8171]
 [ 1.1217]
 [ 1.9924]
 [ 0.8783]
 [ 1.2211]
 [ 3.0844]
 [ 0.6711]
 [ 4.809 ]
 [ 0.782 ]
 [ 1.6667]
 [ 0.5296]
 [ 0.6479]
 [ 1.2933]
 [ 0.6033]
 [ 0.8964]
 [14.0568]
 [ 1.6134]
 [ 1.1659]
 [ 0.7902]
 [ 0.5573]
 [ 0.9168]
 [ 1.1002]
 [ 4.5686]
 [ 3.9976]
 [ 0.8686]
 [ 0.4014]
 [ 4.3851]
 [ 4.4339]
 [ 0.7765]
 [ 1.2943]
 [ 5.1731]
 [ 2.9524]
 [ 3.1681]
 [ 0.6116]
 [ 2.6638]
 [ 0.7507]
 [ 1.0324]
 [ 3.2195]
 [36.3239]
 [ 0.4401]
 [ 4.1938]
 [ 0.7918]
 [38.3294]
 [ 0.996 ]
 [ 0.5479]
 [ 1.0231]
 [ 0.4216]
 [ 1.0179]
 [ 0.5811]
 [11.1049]
 [ 0.6305]
 [ 1.1152]
 [ 1.4102]
 [ 1.0191]
 [ 0.6896]
 [ 0.7975]
 [ 2.5785]
 [ 3.6842]
 [11.508 ]
 [ 0.9276]
 [10.4858]
 [50.5544]
 [ 0.6835]
 [ 2.9058]
 [ 1.3641]
 [ 0.6496]
 [18.8186]
 [ 0.998 ]
 [ 1.674 ]
 [27.4704]
 [ 4.4649]
 [27.8526]
 [ 1.8077]];

obj_multi= [[ 1.8895]
 [18.4741]
 [ 0.9317]
 [ 1.7972]
 [15.9958]
 [ 1.807 ]
 [17.279 ]
 [16.5472]
 [ 2.0846]
 [ 0.3094]
 [ 0.6177]
 [ 0.3356]
 [ 0.3471]
 [ 0.3626]
 [ 0.2529]
 [ 0.4168]
 [41.3595]
 [ 1.2089]
 [ 0.3977]];



%% Plot figure 1
axisticksize = 40;
origin = [200,60];
markersize = 4;
LineWidth = 3;

figure(1)
iters_single = 1:length(obj_single);
hd1 = semilogy(iters_single,obj_single,'bo','MarkerSize',markersize,'MarkerFaceColor','b','LineWidth',LineWidth);
hold on
iters_multi = 1:length(obj_multi);
hd1 = semilogy(iters_multi,obj_multi,'ro','MarkerSize',markersize,'MarkerFaceColor','r','LineWidth',LineWidth);
hold on

idx=find(obj_single==min(obj_single));
hd1 = semilogy(iters_single(idx),obj_single(idx),'bp','MarkerSize',markersize*4,'MarkerFaceColor','b','LineWidth',LineWidth);
hold on
idx=find(obj_multi==min(obj_multi));
hd1 = semilogy(iters_multi(idx),obj_multi(idx),'rp','MarkerSize',markersize*4,'MarkerFaceColor','r','LineWidth',LineWidth);
hold on

gca = get(gcf,'CurrentAxes');
% set(gca,'XTick',[1:length(SOLVE_SLU(:,1))])
% set(gca,'TickLabelInterpreter','none')
% xticklabels(xtick)
% xtickangle(45)
% xlim([0,2.5]);
% ylim([0,2000]);
gca = get(gcf,'CurrentAxes');
set(gca,'DefaultAxesTitleFontWeight','normal');
title('m=4674, n=3608')

legs = {};
legs{1,1} = ['Single-task'];
legs{1,2} = ['Multi-task'];

gca = get(gcf,'CurrentAxes');
set( gca, 'FontName','Times New Roman','fontsize',axisticksize);
str = sprintf('Time (s)');
ylabel(str,'interpreter','Latex')
str = sprintf('Parameter sample index');
xlabel(str,'interpreter','Latex')

gca=legend(legs,'interpreter','none','color','none','NumColumns',1);

set(gcf,'Position',[origin,1000,700]);

fig = gcf;

str = 'history_scalapack.eps';
saveas(fig,str,'epsc')


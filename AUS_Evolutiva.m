function Feat_Index =  AUS_Evolutiva
clear all
global Data 
clear
clc
close all

load AUS.data
Data  = load('AUS.data'); % Se carga el dataset
LargoString =14; % Número de características en el dataset
TamanoTorneo = 2; % Tamaño del torneo
ParametrosGA = gaoptimset('CreationFcn', {@PopFunction},...
                     'PopulationSize',128,...
                     'Generations',50,...
                     'PopulationType', 'bitstring',... 
                     'SelectionFcn',{@selectiontournament,TamanoTorneo},...
                     'MutationFcn',{@mutationuniform, 0.1},...
                     'CrossoverFcn', {@crossoverarithmetic,0.5},...
                     'EliteCount',2,...
                     'StallGenLimit',100,...
                     'PlotFcns',{@gaplotbestf},...  
                     'Display', 'iter'); 
rand('seed',1)
nVars = 14; % Cantidad de variables independientes
FitnessFcn = @FitFunc_SVM; 
format shortg
c = clock;
disp('hora antes = ');
disp(c);
[chromosome,~,~,~,~,~] = ga(FitnessFcn,nVars,ParametrosGA);
format shortg
c = clock;
disp('hora despues = ');
disp(c);
Best_chromosome = chromosome; % Best Chromosome
Feat_Index = find(Best_chromosome==1); % Index of Chromosome
end

%%% Generar la población
function [pop] = PopFunction(LargoString,~,ParametrosGA)
RD = rand;  
pop = (rand(ParametrosGA.PopulationSize, LargoString)> RD); % Initial Population
end

%%% Función Fitness
function [FitVal] = FitFunc_SVM(pop)
    load AUS.data
    individuo=zeros(1,15);
    contador=1;
    for i=1:14
        if pop(1,i)==1
            individuo(1,contador)=i;
            contador=contador+1;
        end
    end
    if contador==1 %% Para evitar que el programa se caiga si el individuo solo contiene ceros
        individuo(1,1)=1;
    end
    ind = individuo(individuo~=0);
    columnasSeleccionadas=AUS(1:end,ind);
    clasificaciones=AUS(1:end,15);

    p=0.5;
    [Train,Test] = crossvalind('HoldOut', clasificaciones,p);

    trainingSample=columnasSeleccionadas(Train,:);
    trainingLabel=clasificaciones(Train,1);
    testingSample=columnasSeleccionadas(Test,:);
    testingLabel=clasificaciones(Test,1);

    svmStruct =svmtrain(trainingSample,trainingLabel,...
        'showplot',false,'kernel_function','rbf','rbf_sigma',0.5);

    outLabel = svmclassify(svmStruct,testingSample,'showplot',false);
    Fitness=sum(grp2idx(outLabel)==grp2idx(testingLabel))/sum(Test)

    FitVal = 1-Fitness;
end
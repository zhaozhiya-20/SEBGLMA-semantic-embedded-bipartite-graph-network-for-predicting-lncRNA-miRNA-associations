load('Known_lncRNA_miRNA_association.mat');
% A=xlsread('E:\论文\LncRNA-miRNA\NDALMA-main\770\dataset\interaction_pair_seq.xlsx');
% A=xlsread('E:\论文\LncRNA-miRNA\case_study\adj\pairs_id.xlsx');

%A=unique(Known_lncRNA_miRNA_association,'rows');
nl=max(A(:,1)); 
nm=max(A(:,2));
[pp,qq]=size(A);
%interaction: adjacency matrix for the disease-microbe association network
%interaction(i,j)=1 means microbe j is related to disease i
for i=1:pp
        inter(A(i,1),A(i,2))=1;
end
save interaction interaction;

gamall=1;
gamamm=1;
%calculate gamad for lncRNA Gaussian kernel calculation
for i=1:nl
    sl(i)=norm(interaction(i,:))^2;
end
gamal=nl/sum(sl')*gamall;
    
%calculate gamal for miRNA Gaussian kernel calculation
for i=1:nm
    sm(i)=norm(interaction(:,i))^2;
end
gamam=nm/sum(sm')*gamamm;
    
%calculate Gaussian kernel for the similarity between lncRNA: kl
for i=1:nl
    for j=1:nl
        lnc_gaussian_similarity_matrix(i,j)=exp(-gamal*(norm(interaction(i,:)-interaction(j,:)))^2);
    end
end
%save lnc_gaussian_similarity_matrix lnc_gaussian_similarity_matrix;

%calculate Gaussian kernel for the similarity between microbe: km
for i=1:nm
    for j=1:nm
        mi_gaussian_similarity_matrix(i,j)=exp(-gamam*(norm(interaction(:,i)-interaction(:,j)))^2);
    end
end 
%save mi_gaussian_similarity_matrix mi_gaussian_similarity_matrix;
       


        
        
        
    
   




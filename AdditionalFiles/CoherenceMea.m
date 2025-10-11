%% compute coherence

function Coh = coherence_3(U,D,kk) % C is word-topic
SS = 0;
for tp = 1:size(U, 2)
    [~,In]=sortrows(U,tp,"descend");
    In_top_w_tp = In(1:kk);
    ss2 = 0;
    for w1=1:kk-1
        for w2 = w1+1:kk
            c1 = sum((D(In_top_w_tp(w1),:) .* D(In_top_w_tp(w2),:)) ~=0);
            c2 = sum(D(In_top_w_tp(w1),:)~=0);
            clg = log((c1+0.01)/c2);
            if isinf(clg) || isnan(clg)
                clg = 0;
            end
            ss2 = ss2 + clg ;
        end
    end

    SS = SS + ss2;

end
Coh = SS/size(U,2) ;
end





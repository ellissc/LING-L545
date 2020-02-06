import sys

line = sys.stdin.readline()

abbrev = ["age.","alb.","alt.","alz.","ang.","ast.","atd.","atp.","aus.","bal.","bas.","btp.","buł.","bůn.","bůł.","cer.","cyg.","cyr.","dia.","dna.","dom.","dow.","dńi.","dům.","důł.","ery.","est.","etc.","fet.","fiń.","gag.","gal.","gas.","gaz.","gen.","grc.","gre.","gyn.","han.","haw.","hel.","hul.","ino.","int.","irl.","itd.","jak.","jap.","jid.","jod.","jům.","kat.","kaz.","kdb.","kjh.","kol.","kor.","krc.","krm.","kůu.","lad.","lap.","las.","lat.","lew.","lit.","los.","lot.","lud.","lŏt.","lůd.","min.","mln.","moc.","mon.","můn.","nah.","nar.","nep.","niy.","nov.","ntp.","okc.","opy.","paś.","per.","poj.","pol.","pop.","por.","pot.","rap.","ras.","ret.","rod.","rok.","ros.","roz.","rum.","rus.","rům.","sch.","sie.","skr.","sua.","szp.","szw.","sōm.","sůl.","tam.","tle.","tmj.","trb.","trl.","tum.","tur.","tyb.","typ.","tys.","tyś.","tyż.","tzn.","tzw.","tło.","uac.","uać.","ukr.","ulg.","uok.","uos.","uot.","uůn.","var.","wal.","woj.","wsi.","wym.","wyn.","wśi.","yng.","zał.","zaś.","zem.","zoc.","zol.","zou.","ông.","čes.","čos.","łac.","łać.","łeb.","ńid.","ńim.","ńym.","ńům.","ůkr.","ůng.","żył.","um.","r.","m.","f.","św.","zm."]
print("start")
while line != '':
    print(line)
    if line.strip() == '':
        line = sys.stdin.readline()
        continue
    for token in line.split(' '):
#        print(token)
#        print(token[-1])
        if token[-1] in '!?':
            sys.stdout.write(token + '\n')
        elif token[-1] == '.':
            if token in abbrev:
                sys.stdout.write(token + ' ')
            else:
                sys.stdout.write(token + '\n')
        else:
            sys.stdout.write(token + ' ')
    line = sys.stdin.readline()

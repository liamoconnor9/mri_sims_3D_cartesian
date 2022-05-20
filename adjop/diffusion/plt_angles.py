import numpy as np
import matplotlib.pyplot as plt

wavenumbers = list(range(1, 40))

angles_T1 = [0.1692949182162829, 0.5086864390609556, 0.8504927890961109, 1.1963566041282592, 1.5479702180101624, 1.9070966972379808, 2.2755920499804714, 2.655428971357784, 3.0487225045341897, 3.4577580164490067, 3.885021904953588, 4.333235468410457, 4.805392372237265, 5.30480013438828, 5.83512600876332, 6.400447558470332, 7.005308053439413, 7.654776566942041, 8.354512234710493, 9.110831513639567, 9.930776345812797, 10.822179781374189, 11.79372369285107, 12.854980547214709, 14.016427598836755, 15.289417158470139, 16.686080715897027, 18.219137834850727, 19.90157358683012, 21.746142402861537, 23.76465448707608, 25.96700798060141, 28.3599522412011, 30.945611868345818, 33.719871886883205, 36.670818068053435, 39.77752382479964, 43.00953856935851, 46.32741178499907]

angles_Tp5 = [0.10081447254655736, 0.3026875023055132, 0.5052942919623882, 0.7091290479431344, 0.9146935732958061, 1.1225004006202186, 1.3330760203026375, 1.5469642328027118, 1.764729654846283, 1.9869614107516764, 2.2142770416524558, 2.4473266672974323, 2.6867974369245458, 2.933418308075103, 3.1879651942406335, 3.4512665246886027, 3.7242092618952958, 4.007745424042284, 4.302899161545617, 4.610774437438316, 4.932563361204231, 5.2695552238639385, 5.623146278133944, 5.994850300427509, 6.386309960217996, 6.799309005423404, 7.235785248058741, 7.6978443000678825, 8.187773961894248, 8.708059102118877, 9.261396780575103, 9.850711253861054, 10.479168354035934, 11.150188540263379, 11.86745767999217, 12.63493431110572, 13.456851759977937, 14.337713035279855, 15.282275881614103]

angles_Tp25 = [0.06012219522217888, 0.1804398474865892, 0.30097751228504144, 0.4218826224859739, 0.5433037514068035, 0.6653910763264363, 0.7882968493907259, 0.9121758780831948, 1.0371860174958285, 1.163488676685813, 1.2912493414882102, 1.42063811624911, 1.55183028700627, 1.6850069088204294, 1.8203554199982424, 1.9580702861781067, 2.0983536773161124, 2.2414161808214925, 2.3874775542466655, 2.5367675211126914, 2.689526613670933, 2.8460070665740944, 3.006473765665067, 3.1712052563041424, 3.3404948158661734, 3.5146515952760904, 3.694001834633723, 3.8788901582235793, 4.069680954329038, 4.266759845470179, 4.470535254746807, 4.681440074036327, 4.899933439743828, 5.126502621669169, 5.361665030264482, 5.60597034711425, 5.86000278279162, 6.124383465301138, 6.399772961023053]

angles_Tp125 = [0.035935186337521526, 0.10782706586725678, 0.1797834987526696, 0.2518476302294105, 0.3240627702380821, 0.3964724599120723, 0.469120538479885, 0.5420512109440849, 0.6153091165601737, 0.688939398338487, 0.7629877736970903, 0.837500606463774, 0.912524980365288, 0.9881087741997311, 1.0643007388580923, 1.1411505763811567, 1.2187090212536653, 1.2970279241046752, 1.376160338047654, 1.4561606078432658, 1.5370844621263486, 1.61898910890476, 1.701933334577998, 1.7859776067148483, 1.8711841808479783, 1.9576172115539432, 2.0453428680924075, 2.1344294549041476, 2.2249475372572713, 2.3169700723768516, 2.4105725463729977, 2.5058331173244905, 2.6028327648790306, 2.701655446736758, 2.8023882624348455, 2.9051216248170633, 3.009949439643647, 3.1169692937700413, 3.2262826523750463]

angles_Tp0625 = [0.021666552351534203, 0.06500604457267702, 0.10836470419706644, 0.15175532600532626, 0.19519072956872824, 0.23868376917526715, 0.28224734386536554, 0.32589440744999065, 0.36963797864093706, 0.4134911512371723, 0.4574671044050146, 0.5015791130652806, 0.545840558398564, 0.5902649384539586, 0.634865878927242, 0.6796571440794829, 0.7246526478175865, 0.7698664649682251, 0.8153128427276719, 0.8610062123427827, 0.9069612009873719, 0.9531926439080438, 0.9997155967915844, 1.0465453484220466, 1.0936974336119465, 1.1411876464411768, 1.189032053800429, 1.2372470092879062, 1.2858491674428616, 1.3348554983573013, 1.384283302677656, 1.4341502270122117, 1.484474279772993, 1.5352738474609056, 1.5865677114322625, 1.6383750651488018, 1.6907155319526161, 1.7436091833759264, 1.7970765580185741]

angles_Tp03125 = [0.013063612741598597, 0.03849179050384738, 0.06406200300193836, 0.08966752380988094, 0.11528672827875724, 0.14092508709326376, 0.1665864812814038, 0.19226274638382543, 0.21797554828961463, 0.2437083137021169, 0.2694794841813498, 0.2952830261213219, 0.3211271254283623, 0.3470132294102117, 0.37294542449050905, 0.39892700411971116, 0.4249614742859089, 0.4510523392654737, 0.4772031227566091, 0.5034173757382163, 0.5296986723686222, 0.5560506127159728, 0.5824768233867841, 0.6089809587893753, 0.6355667023634908, 0.6622377678475865, 0.6889979006131868, 0.7158508790186362, 0.7428005158091021, 0.7698506595291017, 0.7970051959814269, 0.8242680496976722, 0.851643185451273, 0.8791346097860236, 0.9067463725748247, 0.9344825686049972, 0.9623473391981743, 0.9903448738454158, 1.0184794118859304]

plt.scatter(np.array(wavenumbers)**2, np.array(angles_Tp03125) / 0.03125, label=r"$T = 0.03125$")
plt.scatter(np.array(wavenumbers)**2, np.array(angles_Tp0625) / 0.0625, label=r"$T = 0.0625$")
plt.scatter(np.array(wavenumbers)**2, np.array(angles_Tp125) / 0.125, label=r"$T = 0.125$")
plt.scatter(np.array(wavenumbers)**2, np.array(angles_Tp25) / 0.25, label=r"$T = 0.250$")
# plt.scatter(np.array(wavenumbers)**2, np.array(angles_Tp5) / 0.5, label=r"$T = 0.500$")
plt.legend()
plt.title('Gradient Error angle ' + r'$\theta$' + '; L2 Norm')
plt.ylabel(r'$\theta \; / \; T$')
plt.xlabel('Wavenumber Squared')
plt.savefig('diff_angles_L2.png')
plt.show()
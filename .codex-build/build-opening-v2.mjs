import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';

const skillDir='C:/Users/USER/.codex/plugins/cache/openai-primary-runtime/presentations/26.1111.10930/skills/presentations';
const workspaceDir='C:/Users/USER/Desktop/nlp';
const finalPath=path.join(workspaceDir,'作业二','狼人杀身份推断_开题汇报_v3.pptx');
const blue='#00539B',red='#C00000',dark='#111111',font='微软雅黑';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/USER/Desktop/26C02_概率与概率图模型.pptx'));
function tx(s,t,x,y,w,h,size=27,color=dark,bold=false){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{style:'solid',fill:'none',width:0}});a.text=t;a.text.style={typeface:font,fontSize:size,color,bold,verticalAlignment:'middle',autoFit:'shrinkText',wrap:'square',insets:0};return a;}
function head(s,t,n){tx(s,t,58,39,1000,62,37,blue,true);tx(s,String(n),1050,679,25,24,18);}
function row(s,label,detail,y){tx(s,label,66,y,240,57,27,blue,true);tx(s,detail,310,y,735,67,26);}
async function photo(s,file,x,y,w,h,alt){const b=await fs.readFile(path.join(workspaceDir,'作业二','assets',file));s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/jpeg',alt,fit:'cover',position:{left:x,top:y,width:w,height:h}});}

const cover=p.slides.items[0];
tx(cover,'狼人杀玩家身份推断系统',275,318,825,87,50,'#FFFFFF',true);
tx(cover,'根据发言和投票，猜出每个人的隐藏身份',179,420,910,60,30,'#FFFFFF');
tx(cover,'NLP 课程大作业 · 开题汇报',388,497,570,39,23,'#FFFFFF');
cover.speakerNotes.text='本次开题说明：游戏中要判断什么、为什么难、系统准备怎样做。';
const toc=p.slides.items[1];
p.delete(['sh/xcvil0j2','im/98r25g7i','im/n69k36ps','im/lk721w7m','im/zipkzm9g','im/dg72xwra','im/2lczyxon','im/oneh0n6t']);
['游戏与具体例子','我们要解决的问题','已有研究与项目思路','数据和实现步骤','怎样评价结果','预期展示与安排'].forEach((t,i)=>{tx(toc,'□',524,239+i*60,35,38,28,red,true);tx(toc,t,567,238+i*60,500,45,26,i===0?red:dark,true);});
const base=p.slides.items[2];p.delete(['sh/e10f2twf','sh/l4fi9cza','sh/x0vi5czy']);const blank=base.copy();for(let i=0;i<9;i++)p.slides.addCopy(blank);const S=p.slides.items.slice(2);

// 3: Game introduction with authentic illustrative photo.
{
 const s=S[0];head(s,'游戏背景：大家讨论，试着找出隐藏的狼人',3);
 tx(s,'每个人拿到一个不公开的身份。',66,174,540,67,29,blue,true);
 tx(s,'少数人是狼人，其余人要通过讨论找出他们。',66,250,540,91,28);
 tx(s,'大家每轮发言，再投票选出一个人；之后还会出现新线索。',66,365,540,112,27);
 tx(s,'发言可能是真话，也可能是为了误导别人。',66,518,540,74,27,red,true);
 await photo(s,'students_board_game.jpg',652,182,398,313,'大学生围桌进行桌游的真实照片，仅作游戏讨论情境示意');
 tx(s,'桌游讨论情境示意',713,510,330,32,18,'#555555');
 s.speakerNotes.text='照片来源：Evan Marvell / Unsplash，https://unsplash.com/photos/people-playing-a-board-game-at-a-table-indoors-aHketkfrSFg 。照片不是本项目的狼人杀数据。';
}
// 4: Concrete example.
{
 const s=S[1];head(s,'具体例子：只听发言，可能会判断错',4);
 tx(s,'第一轮',68,171,166,46,26,blue,true);tx(s,'3号说：“7号很可疑，我会投给7号。”',236,171,785,62,28);
 tx(s,'投票时',68,268,166,46,26,blue,true);tx(s,'3号实际投给了5号。',236,268,785,62,28);
 tx(s,'第二轮',68,365,166,46,26,blue,true);tx(s,'3号又说：“我一直觉得7号没问题。”',236,365,785,62,28);
 tx(s,'该怎么判断？',68,485,230,55,27,red,true);
 tx(s,'先看他说的话与投票是否一致，再看其他人的反应。\n矛盾是一条线索，不能直接断定他是狼人。',304,474,482,145,25);
 await photo(s,'playing_cards.jpg',809,470,223,149,'真实纸牌游戏照片，仅作隐藏信息情境示意');
 tx(s,'纸牌游戏情境示意',814,629,220,28,17,'#555555');
 s.speakerNotes.text='例子为虚构，用来解释任务，不是实验数据。照片来源：Bea Vallejo / Unsplash，https://unsplash.com/photos/people-playing-cards-at-a-table-rRwFQIwZ1Ig 。照片不是狼人杀对局。';
}
// 5: Task.
{
 const s=S[2];head(s,'我们的任务：猜出每个人的具体身份',5);
 row(s,'只看什么？','当时大家都能听到的发言、投票和公开结果。',177);
 row(s,'猜出什么？','每名玩家是狼人、普通玩家，还是有特殊能力的角色。',290);
 row(s,'怎么展示？','给出当前判断，并指出依据来自哪句话或哪次投票。',403);
 tx(s,'例如：第二天投票结束后，用前两天的公开记录判断所有玩家。',68,566,985,72,27,red,true);
 s.speakerNotes.text='真实身份用于事后核对，不会提前告诉模型。首版角色种类取决于最终选定的数据和规则。';
}
// 6: Related work.
{
 const s=S[3];head(s,'已有研究做到了什么？我们还要做什么？',6);
 row(s,'早期研究','能从语言中判断谁更像在隐瞒身份；多是两类判断。',174);
 row(s,'长对话研究','证明前后发言和游戏进展有用，但研究的游戏规则不同。',281);
 row(s,'狼人杀智能体','重点让模型学会参与游戏；我们重点判断所有人的身份。',388);
 tx(s,'本项目要把公开记录整理、逐轮判断、证据回查和结果评价做成一个系统。',69,548,985,91,25,red,true);
 s.speakerNotes.text='代表来源：Girlea et al. 2016 https://aclanthology.org/N16-1047/ ；Ibraheem et al. 2022 https://aclanthology.org/2022.naacl-main.11/ ；Stepputtis et al. 2023 https://aclanthology.org/2023.findings-emnlp.748/ ；MaKTO https://arxiv.org/abs/2501.14225 ；CSP4SDG https://arxiv.org/abs/2511.06175 。CSP4SDG已研究身份推断，此处不宣称任务首创。已有工作的范围差异不等于其性能缺陷。';
}
// 7: Approach.
{
 const s=S[4];head(s,'基本思路：把每个人说的和做的连起来看',7);
 row(s,'记住发言','记录谁说自己是什么身份、怀疑谁、后来有没有改口。',169);
 row(s,'核对行为','把说法与实际投票、公开结果放在一起比较。',274);
 row(s,'结合规则','每种身份的人数有限；最终结果要符合人数要求。',379);
 row(s,'保留依据','每个判断能回到原来的发言或投票记录。',484);
 s.speakerNotes.text='玩家说自己是什么，不代表其身份得到确认；投票也不能单独说明谁是狼人。';
}
// 8: Pipeline.
{
 const s=S[5];head(s,'实现步骤：从对局记录到身份判断',8);
 const a=[['1  整理记录','保留公开发言、投票与结果，排除秘密信息。'],['2  按时间截取','要判断第二天，就只使用第二天之前的记录。'],['3  交给模型','让中文大语言模型根据当前记录猜每个人的身份。'],['4  检查与展示','核对身份人数，展示依据和每轮判断的变化。']];
 a.forEach((r,i)=>row(s,r[0],r[1],166+i*113));
 tx(s,'先做一个能运行的版本，再检验整理历史发言是否能提高效果。',68,632,985,47,23,red,true);
 s.speakerNotes.text='拟使用Qwen3-8B。是否进行额外模型训练由数据量、算力及基线效果决定。';
}
// 9: Data.
{
 const s=S[6];head(s,'数据准备：模型只能看到当时公开的信息',9);
 tx(s,'候选数据：MaKTO-Werewolf 的公开对局记录',68,165,985,58,28,blue,true);
 tx(s,'可以给模型看',70,270,445,55,27,blue,true);tx(s,'不能提前给模型看',562,270,490,55,27,red,true);
 tx(s,'玩家说的话\n大家的投票\n主持人公布的结果',70,340,435,188,28);
 tx(s,'每个人的真实身份\n夜间只有少数人知道的行动\n判断时点之后发生的事',562,340,488,188,28);
 tx(s,'开题后先检查哪些对局记录完整，再决定正式使用的数据范围。',68,585,985,66,25,red,true);
 s.speakerNotes.text='候选来源：https://huggingface.co/datasets/ReneeYe/werewolf_game_reasoning 。目前未核验可用对局数；部分样例有私人信息，需要过滤。';
}
// 10: Evaluation.
{
 const s=S[7];head(s,'怎样判断系统有没有用？',10);
 row(s,'对照一','按照每种身份的人数随机猜。',166);
 row(s,'对照二','让模型直接读全部可用的公开记录。',264);
 row(s,'我们的方案','先整理每个人的历史发言和投票，再让模型判断。',362);
 tx(s,'比较每种身份猜对了多少、是否能猜对整局，以及不同轮次的变化。',69,511,986,98,27,blue,true);
 tx(s,'同一局的各轮记录放在同一组，避免提前“见过答案”。',69,612,985,50,23,red,true);
 s.speakerNotes.text='正式指标包括具体角色Macro-F1、整局完全正确率和分轮次表现。当前没有实验成绩，开题PPT不展示虚构数字。';
}
// 11: Expected output.
{
 const s=S[8];head(s,'最后希望做出什么？',11);
 row(s,'选择一局','选定第几天，查看系统当时能知道的记录。',166);
 row(s,'查看身份表','看到每个人的身份猜测和引用的发言或投票。',281);
 row(s,'比较变化','看看新一轮信息出现后，哪些判断发生改变。',396);
 tx(s,'另外提交实验对比、各类身份的识别结果和判断错误的例子。',69,567,985,70,26,red,true);
 s.speakerNotes.text='此页描述预期成果，系统与实验尚未完成。信息不足时允许列出候选身份。';
}
// 12: Plan.
{
 const s=S[9];head(s,'计划与可行性',12);
 row(s,'第一步','检查数据，确认游戏规则，提取公开记录。',167);
 row(s,'第二步','建立第一个模型版本，完成身份判断与证据展示。',268);
 row(s,'第三步','比较不同方法，分析错误，再完善演示系统。',369);
 tx(s,'要重点处理的困难',67,496,260,49,27,red,true);
 tx(s,'记录里混入秘密信息 → 严格筛选并人工抽查\n可用对局不够 → 选择记录较完整的一种玩法\n有些身份难判断 → 保留备选并如实报告',69,552,950,114,24);
 s.speakerNotes.text='开题阶段确认的是路线；数据规模和模型效果还需要后续实测。';
}

await fs.mkdir(path.join(workspaceDir,'.codex-finalizer'),{recursive:true});await fs.mkdir(path.dirname(finalPath),{recursive:true});
const candidatePath=path.join(workspaceDir,'.codex-finalizer','opening-v3-candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(skillDir,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1111-570-0_26.1111.570.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe',integrityValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','10691495,7559675','--validate-heading-fit'],sourceTemplatePath:'C:/Users/USER/Desktop/26C02_概率与概率图模型.pptx',verifyArtifactToolImport:true,receiptPath:path.join(workspaceDir,'.codex-finalizer','opening-v3-validation.json')});
console.log(JSON.stringify({slides:p.slides.items.length,finalPath,layout:result.presentationLayout,integrity:result.packageIntegrity},null,2));

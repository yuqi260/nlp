import fs from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';

const root='C:/Users/USER/Desktop/nlp/作业二';
const iconRoot=path.join(root,'assets','role-icons');
const out=path.join(root,'九人狼人杀身份配置_网易角色版');
const blue='#07549C',red='#A33742',ink='#192B42',grey='#526176';
const esc=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;');
const text=(s,x,y,size=24,color=ink,weight=400,anchor='start')=>`<text x="${x}" y="${y}" fill="${color}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${esc(s)}</text>`;
async function icon(name,x,y,size,color){
 let raw=await fs.readFile(path.join(iconRoot,name+'.svg'),'utf8');
 let content=raw.replace(/^[\s\S]*?<svg[^>]*>/,'').replace(/<\/svg>\s*$/,'');
 if(name==='wolf-head'){
  content=content.replace(/<path d="M0 0h512v512H0z"\s*\/>/,'').replaceAll('fill="#fff"',`fill="${color}"`);
  return `<svg x="${x}" y="${y}" width="${size}" height="${size}" viewBox="0 0 512 512">${content}</svg>`;
 }
 return `<svg x="${x}" y="${y}" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${content}</svg>`;
}
async function character(code,x,y,w,h){
 const file=path.join(root,'assets','netease-roles',code+'.png');
 const bytes=await sharp(file).trim().png().toBuffer();
 return `<image x="${x}" y="${y}" width="${w}" height="${h}" preserveAspectRatio="xMidYMax meet" href="data:image/png;base64,${bytes.toString('base64')}"/>`;
}

let b=[];
b.push(`<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080"><style>text{font-family:'Microsoft YaHei','微软雅黑',sans-serif}</style><rect width="1920" height="1080" fill="white"/>`);
b.push(text('九人狼人杀 · 身份配置',80,94,52,blue,700));
b.push(text('预言家 1 人  ·  女巫 1 人  ·  猎人 1 人  ·  平民 3 人  ·  狼人 3 人',82,150,26,grey));
b.push(`<path d="M80 187H1840" stroke="${blue}" stroke-width="4"/>`);
b.push('<rect x="80" y="225" width="1160" height="700" rx="18" fill="#F1F6FC"/><rect x="1272" y="225" width="568" height="700" rx="18" fill="#FAF0F1"/>');
b.push(text('好人阵营',120,285,34,blue,700),text('6 人',1198,285,32,blue,700,'end'));
b.push(text('狼人阵营',1312,285,34,red,700),text('3 人',1798,285,32,red,700,'end'));
const roles=[
 {name:'预言家',image:'yyj',cx:300,lines:['每晚可查验一名玩家','得知其是好人还是狼人']},
 {name:'女巫',image:'nw',cx:660,lines:['拥有一瓶解药','和一瓶毒药']},
 {name:'猎人',image:'lr',cx:1020,lines:['符合条件出局时','可开枪带走一人']}
];
for(const r of roles){
 b.push(await character(r.image,r.cx-120,315,240,210));
 b.push(text(r.name+' × 1',r.cx,567,34,blue,700,'middle'));
 r.lines.forEach((l,i)=>b.push(text(l,r.cx,606+i*35,24,ink,400,'middle')));
}
b.push('<path d="M120 676H1200" stroke="#CCDCEC" stroke-width="2"/>');
b.push(text('平民 × 3',120,753,31,blue,700),text('没有特殊能力',120,801,24,ink),text('依靠发言与投票判断',120,841,24,ink));
for(const cx of [600,820,1040]){b.push(await character('pm',cx-75,700,150,165));b.push(text('平民',cx,902,25,blue,600,'middle'));}
for(const cx of [1380,1556,1732])b.push(await character('langr',cx-76,333,152,270));
b.push(text('狼人 × 3',1556,653,38,red,700,'middle'));
b.push(text('夜间共同选择袭击目标',1556,706,25,ink,400,'middle'),text('白天隐藏身份、参与讨论',1556,748,25,ink,400,'middle'));
b.push('<path d="M1312 786H1800" stroke="#E5C7CC" stroke-width="2"/>');
b.push(text('同属一个阵营',1556,839,26,red,600,'middle'),text('通常互相知道身份',1556,881,25,ink,400,'middle'));
b.push(text('所有角色都会发言和投票；公开声明的身份不一定是真实身份。',80,985,27,blue,600));
b.push(text('注：女巫用药限制、猎人技能触发条件等，以具体对局规则为准。',80,1034,19,grey));
b.push(text('角色立绘：网易《狼人杀》官网；素材版权归原权利人所有。',1840,1063,15,grey,400,'end'));
b.push('</svg>');
const svg=b.join('\n');
await fs.writeFile(out+'.svg',svg,'utf8');
await sharp(Buffer.from(svg)).resize(2880,1620).png().toFile(out+'.png');
await fs.writeFile(out+'_来源.txt',`角色配置：按用户指定的9人配置绘制。\n角色立绘来源：网易《狼人杀-官方正版》官方网站 https://langrensha.com/\n官网角色数据：https://langrensha.res.netease.com/pc/gw/20190509150909/js/kapaiData_02806e2.js\n预言家：https://langrensha.res.netease.com/pc/gw/20190509150909/data/kapai_peo/yyj.png\n女巫：https://langrensha.res.netease.com/pc/gw/20190509150909/data/kapai_peo/nw.png\n猎人：https://langrensha.res.netease.com/pc/gw/20190509150909/data/kapai_peo/lr.png\n平民：https://langrensha.res.netease.com/pc/gw/20190509150909/data/kapai_peo/pm.png\n狼人：https://langrensha.res.netease.com/pc/gw/20190509150909/data/kapai_peo/langr.png\n规则参考：https://langrensha.com/wanfa/guize/2017/10/18/26899_719311.html\n角色素材版权归网易及原权利人所有，用于本次课程汇报说明。仅移除空白透明边界并按比例缩放，未使用AI重绘。\n`,'utf8');
console.log(JSON.stringify({png:out+'.png',svg:out+'.svg',size:[2880,1620]}));

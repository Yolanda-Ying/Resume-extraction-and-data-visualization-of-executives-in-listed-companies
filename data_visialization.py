from py2neo import Graph, Node, Relationship
import csv

if __name__ == "__main__":
    # 连接数据库
    graph = Graph("http://localhost:7474", username="neo4j", password='YolandaYing')
    # 文件流
    csv_file  = csv.reader(open('FinalResult_aftercleaning.csv','r',encoding='utf-8'))
    # 分类节点
    for line in csv_file:
        # 跳过第一行
        if line[0] == '股票代码':
            continue
        elif line[0] == 'Stock':
                continue
        else:

            #------------------------------------------创建实体对象--------------------------------------------------#

            #创建高管对象

            Person = Node('高管',Name = line[2],Duty = line[3],Gender = line[4],Birthday = line[5],Nationality = line[6],
                          EducationLevel = line[7],Honer = line[8],Major = line[10])

            #创建公司对象
            if(line[9] == ''):
                print(line[0])
            else:
                for com in line[9].split(';'):
                    Company = Node('公司',ParttimeCompany = com)

            #创建学校对象
            if (line[11] == ''):
                print(line[0])
            else:
                for sch in line[11].split(';'):
                    School = Node('学校', SchoolName=sch)


            #创建股票对象
            Stock = Node('股票',StockNumber = line[0],StockName = line[1])

            # ------------------------------------------创建实体关系对象--------------------------------------------------#

            BelongtoListedCompany = Relationship(Person,'所属上市公司',Stock)
            StudyIn = Relationship(Person, '就读于', School)
            ServeIn = Relationship(Person, '兼职公司', Company)

            graph.create(Person)
            graph.create(Company)
            graph.create(School)
            graph.create(Stock)
            graph.create(BelongtoListedCompany)
            graph.create(StudyIn)
            graph.create(ServeIn)
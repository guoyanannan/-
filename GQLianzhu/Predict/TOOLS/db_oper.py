import pymssql
from TOOLS.file_oper import Print


def ConnectDB(CamNum,DefectDBIP):
    global connSet
    connSet=[]
    for i in range(CamNum):
        DataBaseName="ClientDefectDB{}".format(i+1)
        conn=pymssql.connect(DefectDBIP,"ARNTUSER","ARNTUSER",DataBaseName)
        connSet.append(conn)

def CreateAllCamProcedure(CamNum,DefectDBIP):
    for CamNo in range(1,CamNum+1):
        DataBaseName="ClientDefectDB{}".format(CamNo)
        with pymssql.connect(DefectDBIP,"ARNTUSER","ARNTUSER",DataBaseName) as conn:
            with conn.cursor(as_dict=True) as cursor:
                Print("{} Create ProCedure Success!".format(DataBaseName))
                cursor.execute('''
                    IF EXISTS (SELECT * FROM DBO.SYSOBJECTS WHERE ID = OBJECT_ID(N'[dbo].[AddClassifiedDefectToTempTable]') and OBJECTPROPERTY(ID, N'IsProcedure') = 1)
                    DROP PROCEDURE [dbo].[AddClassifiedDefectToTempTable]
                ''')
                
                cursor.execute('''
                CREATE PROCEDURE AddClassifiedDefectToTempTable
                    @DefectNo int,
                    @SteelNo int,
                    @CameraNo smallint,
                    @ImageIndex smallint,
                    @Class smallint,
                    @Grade smallint,
                    @LeftInImg smallint,
                    @RightInImg smallint,
                    @TopInImg smallint,
                    @BottomInImg smallint,
                    @LeftInSteel smallint,
                    @RightInSteel smallint,
                    @TopInSteel int,
                    @BottomInSteel int,
                    @Area int,
                    @Cycle smallint
                AS BEGIN
                    insert into DefectTempClassified (DefectNo,SteelNo,CameraNo,ImageIndex,Class,Grade,LeftInImg,RightInImg,TopInImg,BottomInImg,LeftInSteel,RightInSteel,TopInSteel,BottomInSteel,Area,Cycle,ImgData) values (@DefectNo,@SteelNo,@CameraNo,@ImageIndex,@Class,@Grade,@LeftInImg,@RightInImg,@TopInImg,@BottomInImg,@LeftInSteel,@RightInSteel,@TopInSteel,@BottomInSteel,@Area,@Cycle,NULL)
                END
                ''')
            conn.commit()

def WriteDatabase(CamNo,DefectDBIP,DefectList):
    if len(DefectList[CamNo-1])>0:
        DataBaseName="ClientDefectDB{}".format(CamNo)
        with pymssql.connect(DefectDBIP,"ARNTUSER","ARNTUSER",DataBaseName) as conn:
            with conn.cursor(as_dict=True) as cursor:
                for i in range(len(DefectList[CamNo-1])):                  
                    #if int(DefectList[CamNo-1][i][4]) in EnableClassDetectNo:#屏蔽不关注类别
                    cursor.callproc('AddClassifiedDefectToTempTable',(int(DefectList[CamNo-1][i][0]),int(DefectList[CamNo-1][i][1]),int(DefectList[CamNo-1][i][2]),int(DefectList[CamNo-1][i][3]),
                                                                  int(DefectList[CamNo-1][i][4]),int(DefectList[CamNo-1][i][5]),int(DefectList[CamNo-1][i][6]),int(DefectList[CamNo-1][i][7]),
                                                                  int(DefectList[CamNo-1][i][8]),int(DefectList[CamNo-1][i][9]),int(DefectList[CamNo-1][i][10]),int(DefectList[CamNo-1][i][11]),
                                                                  int(DefectList[CamNo-1][i][12]),int(DefectList[CamNo-1][i][13]),int(DefectList[CamNo-1][i][14]),int(DefectList[CamNo-1][i][15])))
                    Print("Normal>>成功写入数据库1条记录")
            conn.commit()
        DefectList[CamNo-1]=[]

⏺ 用 superuser 先创建用户再建库：

  -- 以 superuser 身份执行
  CREATE USER holyeval_user_prod WITH PASSWORD 'your_password';
  CREATE DATABASE holyeval_prod OWNER holyeval_user_prod;

  如果当前连接用户不是 superuser，需要先授权：

  -- 让当前用户（如 postgres）能代理该角色
  GRANT holyeval_user_prod TO current_user;
  CREATE DATABASE holyeval_prod OWNER holyeval_user_prod;

  或者直接用 postgres superuser 连接后执行。

❯ 写成文档 

⏺ PostgreSQL 数据库准备（生产环境）

  前提条件

  需以 superuser（如 postgres）身份连接 PostgreSQL 实例执行以下操作。

  psql -h your-pg-host -U postgres

  操作步骤

  -- 1. 创建用户
  CREATE USER holyeval_user_prod WITH PASSWORD 'your_password';

  -- 2. 创建数据库，指定 owner
  CREATE DATABASE holyeval_prod OWNER holyeval_user_prod;

  对应环境变量

  PG_HOST_EVAL=your-pg-host
  PG_PORT_EVAL=5432
  PG_USER_EVAL=holyeval_user_prod
  PG_PASSWORD_EVAL=your_password
  PG_DBNAME_EVAL=holyeval_prod
  PG_SCHEMA_EVAL=holyeval

  说明

  - 必须使用 superuser 执行，普通用户无法将数据库 owner 设为自身以外的角色。
  - Schema holyeval 无需手动创建，hma-web 启动时 prisma migrate deploy 会自动创建 schema 并执行迁移。
  - 环境变量由远程配置中心（CONFIG_SERVER）下发，不直接写入代码仓库。

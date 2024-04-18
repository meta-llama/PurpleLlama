// License: MIT (c) JS Foundation and other contributors, https://js.foundation

window.testEval = function (name) {
    window.globalName = name;
    // ruleid: eval-with-expression
    eval(`console.log("${name}");`);
    // ruleid: eval-with-expression
    eval(name);
    //ruleid: eval-with-expression
    eval("console.log('" + name + "');");

    // Direct call to eval, uses local scope
    // ok: eval-with-expression
    eval("console.log(name);");

    // Indirect Call to eval, only accesses global vars
    // This looks like a harmless string but contains user input variable data that gets evaluated.
    //ruleid: eval-with-expression
    (0, eval)(`console.log("${name}");`);

    // ok: eval-with-expression
    (0, eval)(`console.log(globalName);`);

    // call through alias
    //ruleid: eval-with-expression
    var foo = eval;
    foo(`console.log("${name}");`);

    // ok: eval-with-expression
    var foo = eval;
    foo(`console.log(globalName);`);

    //ruleid: eval-with-expression
    this.eval("console.log('" + name + "');");

    // ok: eval-with-expression
    this.eval("console.log(globalName);");

    // Indirect call through member access
    const obj = { eval };
    //ruleid: eval-with-expression
    obj.eval("console.log('" + name + "');");

    // Indirect call through conditional operator
    //ruleid: eval-with-expression
    eval?.(`console.log("${name}");`);

    //ok: eval-with-expression
    eval?.(`console.log(globalName);`);
  };

  window.testSetTimeout = function (name) {
    window.myname = name;

    // Exploit using http://localhost:3000/dangerous-sinks?name=hi%27);print();//
    // ruleid: eval-with-expression
    setTimeout(`console.log('${name}');`, 1000);

    // ok: eval-with-expression
    setTimeout("console.log(myname);", 1000);

    // ok: eval-with-expression
    setTimeout(function () {
      console.log("setTimeout Safe Sink 2 : " + name);
    }, 500);

    // Passing user input to setTimeout
    const userScript = function () {
      console.log("setTimeout Safe Sink 3 : " + name);
    };

    // ok: eval-with-expression
    setTimeout(userScript, 500);

    // Exploit using http://localhost:3000/dangerous-sinks?name=hi%22);print();//
    // ruleid: eval-with-expression
    setTimeout("console.log('setTimeout Sink 5 : " + name + "');", 1000);

    window.x = 10;
    window.y = 22;
    // this will evaluate, but variables in this format can never break out of the code.
    // ok: eval-with-expression
    setTimeout("console.log('setTimeout Sink 6 :' + (x+y));", 1000);

    // ok: eval-with-expression
    setTimeout("console.log(x+y);", 1000);
  };

  window.testSetInterval = function (name) {
    window.myname = name;

    // Exploit using http://localhost:3000/dangerous-sinks?name=hi%27);print();//
    // ruleid: eval-with-expression
    window.intervalId1 = setInterval(`console.log('${name}');`, 1000);

    // ok: eval-with-expression
    window.intervalId2 = setInterval("console.log(myname);", 1000);

    // ok: eval-with-expression
    window.intervalId3 = setInterval(function () {
      console.log("setTimeout Safe Sink 2 : " + name);
    }, 500);

    // Passing user input to setTimeout
    const userScript = function () {
      console.log("setTimeout Safe Sink 3 : " + name);
    };

    // ok: eval-with-expression
    window.intervalId4 = setInterval(userScript, 500);

    // Exploit using http://localhost:3000/dangerous-sinks?name=hi%22);print();//
    // ruleid: eval-with-expression
    window.intervalId5 = setInterval(
      "console.log('setTimeout Sink 5 : " + name + "');",
      1000
    );

    window.x = 10;
    window.y = 22;
    // ok: eval-with-expression
    window.intervalId6 = setInterval(
      "console.log('setTimeout Sink 6 :' + (x+y));",
      1000
    );
  };

  window.testFunctionConstructor = function (name) {
    // To exploit - http://localhost:3000/dangerous-sinks?name=hi%22;alert(1);var%20a=%221
    // ruleid: eval-with-expression
    const sayHello = new Function(
      `return function (name) { var str = "Hello, ${name}"; return str; }`
    )();
    console.log(sayHello(name));

    // To exploit - http://localhost:3000/dangerous-sinks?name=%22%27;alert(1);//
    // ruleid: eval-with-expression
    const userFormula = "var str = '" + name + "'; return str;";
    const sum = Function(userFormula)();
    console.log(sum);

    const userFormula2 =
      "return function (a) {var str = 'Hello, ' + a; return str;}";
    // ok: eval-with-expression
    const sum2 = new Function("a", userFormula2)();
    console.log(sum2(name));

    // ok: eval-with-expression
    const sayHello2 = Function(
      "return function (name) { return `Hello, ${name}` }"
    )();
    // call the function
    console.log(sayHello2(name));

    // To exploit: http://localhost:3000/dangerous-sinks?name=hi%22);print();//
    // ruleid: eval-with-expression
    const userData = `console.log("Function constructor call - ${name}");`;
    console.log(Function(userData)());

    // ruleid: eval-with-expression
    const userData2 = `console.log("Function constructor call - ${name}");`;
    console.log(new Function(userData2)());
  };
